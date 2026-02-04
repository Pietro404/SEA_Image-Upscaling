#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <string.h>

// Include STB image libraries
// Rimuove warnings
#pragma nv_diag_suppress 550
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Costanti per Tiling
#define TILE_W 16
#define TILE_H 16
#define SHARED_DIM 32

// Fornisce file, riga, codice e descrizione dell'errore
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}
// Funzione timer CPU: misura il tempo wall-clock visto dalla CPU
// Imprecisa, genera overhead, deprecata, solo a scopo didattico
double cpuSecond() {
      struct timespec ts;
      timespec_get(&ts, TIME_UTC);
      return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

// PARAMETRI IN CONSTANT MEMORY
// Raggruppati parametri costanti per ridurre uso dei registri e per
// sfruttare cache delle costanti (veloce per accesso broadcast)
struct KernelParams {
    int width;
    int height;
    int new_width;
    int new_height;
    int channels;
    float x_ratio;
    float y_ratio;
};

// Dichiarazione variabile Constant Memory (Global Scope)
__constant__ KernelParams c_params;


// Nearest Neighbor (Constant + Shared)
__global__ void nn_kernel_const_shared(unsigned char *input, unsigned char *output) {
    // Accesso diretto a c_params senza passarli come argomenti
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Coordinate di base per il caricamento della tile: angolo del blocco output mappato sullo spazio input
	// (angolo in alto a sx della zona di input necessaria al blocco)
    int base_src_x = (int)(blockIdx.x * blockDim.x * c_params.x_ratio);
    int base_src_y = (int)(blockIdx.y * blockDim.y * c_params.y_ratio);
    
	// Caricamento in shared memory
    // Allocazione statica: [Y][X][Canali RGB]
	// Riduzione della Latenza: s_input risiede nella cache on-chip (L1/Shared), molto più veloce della DRAM globale
    __shared__ unsigned char s_input[SHARED_DIM][SHARED_DIM][3];
    
	// Thread linearizzati per il caricamento cooperativo
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
	
    // Ogni thread carica uno o più pixel nella shared memory finché non riempiamo SHARED_DIM x SHARED_DIM
    // Nota: in upscaling l'area utile reale è piccola ma si carica una regione fissa per semplicità e sicurezza
    for (int i = tid; i < SHARED_DIM * SHARED_DIM; i += num_threads) {
        int s_r = i / SHARED_DIM; // riga in shared
        int s_c = i % SHARED_DIM; // colonna in shared
		
		// Coordinate globali corrispondenti nell'immagine di Input + clamp manuale
        int gx = min(base_src_x + s_c, c_params.width - 1);
        int gy = min(base_src_y + s_r, c_params.height - 1);
		
        // Lettura coalesced (se possibile) e scrittura in shared
        int g_idx = (gy * c_params.width + gx) * c_params.channels;
		
        // Unrolling manuale dei canali per evitare loop interni nel caricamento
        if (c_params.channels == 3) {
            s_input[s_r][s_c][0] = input[g_idx + 0];
            s_input[s_r][s_c][1] = input[g_idx + 1];
            s_input[s_r][s_c][2] = input[g_idx + 2];
        }
    }
	
	// Barriera per attesa che tutti i thread abbiano finito caricamento cache
    __syncthreads();
	
	// Check limiti output
    if (x >= c_params.new_width || y >= c_params.new_height) return;
	
	// Logica NN: prende il pixel più vicino
    int src_x = (int)(x * c_params.x_ratio);
    int src_y = (int)(y * c_params.y_ratio);
    
	// Calcolo indici relativi alla tile specifica in shared memory (mapping)
    // La shared memory inizia da (base_src_x, base_src_y) (relative coordinate)
    int s_x = src_x - base_src_x;
    int s_y = src_y - base_src_y;
	
    // Clamp per sicurezza (evita fuori intervallo shared)
    s_x = max(0, min(s_x, SHARED_DIM - 1));
    s_y = max(0, min(s_y, SHARED_DIM - 1));

    for (int c = 0; c < c_params.channels; c++) {
        output[(y * c_params.new_width + x) * c_params.channels + c] = s_input[s_y][s_x][c];
    }
}

// Bilineare (Constant + Shared)
__global__ void bilinear_kernel_const_shared(unsigned char *input, unsigned char *output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	
    // Coordinate di base per il caricamento della tile: angolo del blocco output mappato sullo spazio input
	// (angolo in alto a sx della zona di input necessaria al blocco)
    int base_src_x = (int)((blockIdx.x * blockDim.x) * c_params.x_ratio);
    int base_src_y = (int)((blockIdx.y * blockDim.y) * c_params.y_ratio);
    
	// Caricamento in shared memory
    // Allocazione statica: [Y][X][Canali RGB]
	// Riduzione della Latenza: s_input risiede nella cache on-chip (L1/Shared), molto più veloce della DRAM globale
    __shared__ unsigned char s_input[SHARED_DIM][SHARED_DIM][3];

	// Thread linearizzati per il caricamento cooperativo
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
	
    for (int i = tid; i < SHARED_DIM * SHARED_DIM; i += num_threads) {
		int s_r = i / SHARED_DIM; // riga in shared
        int s_c = i % SHARED_DIM; // colonna in shared
		
		// Coordinate globali corrispondenti nell'immagine di Input + clamp manuale
        int gx = min(base_src_x + s_c, c_params.width - 1);
        int gy = min(base_src_y + s_r, c_params.height - 1);
		
        // Lettura coalesced (se possibile) e scrittura in shared
        int g_idx = (gy * c_params.width + gx) * c_params.channels;
		
        // Unrolling manuale dei canali per evitare loop interni nel caricamento
        if (c_params.channels == 3) {
            s_input[s_r][s_c][0] = input[g_idx + 0];
            s_input[s_r][s_c][1] = input[g_idx + 1];
            s_input[s_r][s_c][2] = input[g_idx + 2];
        }		
    }
    __syncthreads();

    // Check limiti output
    if (x >= c_params.new_width || y >= c_params.new_height) return;
    
	// Coordinate float nello spazio input
    float gx = x * c_params.x_ratio;
    float gy = y * c_params.y_ratio;
    
	int x0 = (int)gx;
    int y0 = (int)gy;
    
	float dx = gx - x0;
    float dy = gy - y0;
	
    // Calcolo indici relativi alla tile specifica in shared memory (mapping)
    // La shared memory inizia da (base_src_x, base_src_y) (relative coordinate)
    int s_x0 = x0 - base_src_x;
    int s_y0 = y0 - base_src_y;
	
	// Clamp manuale per sicurezza sui bordi del tile
    int sx0_cl = max(0, min(s_x0, SHARED_DIM - 2)); 
    int sy0_cl = max(0, min(s_y0, SHARED_DIM - 2));

    for (int c = 0; c < c_params.channels; c++) {
        float p00 = s_input[sy0_cl][sx0_cl][c];
        float p10 = s_input[sy0_cl][sx0_cl + 1][c]; // usa x1
        float p01 = s_input[sy0_cl + 1][sx0_cl][c]; // usa y1
        float p11 = s_input[sy0_cl + 1][sx0_cl + 1][c]; // usa x1 e y1

        float value = p00 * (1 - dx) * (1 - dy) +
                      p10 * dx * (1 - dy) +
                      p01 * (1 - dx) * dy +
                      p11 * dx * dy;

        output[(y * c_params.new_width + x) * c_params.channels + c] = (unsigned char)value;
    }
}

__device__ float cubic(float x) {
    const float a = -0.5f;
    x = fabsf(x);
    if (x <= 1.0f) return (a + 2) * x * x * x - (a + 3) * x * x + 1;
    else if (x < 2.0f) return a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a;
    else return 0.0f;
}

// Bicubica (Constant + Shared)
__global__ void bicubic_kernel_const_shared(unsigned char *input, unsigned char *output) {
    // Coordinate globali del pixel di output
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	
    // Identificazione area input ROI (Region of Interest):
    // calcolo di quale porzione dell'immagine di input serve a questo* blocco di thread
    // (trovare il pixel input corrispondente all'angolo in alto a sinistra del blocco output)
    int start_out_x = blockIdx.x * blockDim.x;
    int start_out_y = blockIdx.y * blockDim.y;
	
    // Coordinate di base per il caricamento della tile: angolo del blocco output mappato sullo spazio input
	// (angolo in alto a sx della zona di input necessaria al blocco)
    int base_src_x = (int)(start_out_x * c_params.x_ratio);
    int base_src_y = (int)(start_out_y * c_params.y_ratio);

    // Origine indietro di 1 per coprire l'Halo sinistro/superiore (necessario per bicubica m=-1)
    base_src_x -= 1;
    base_src_y -= 1;
    
	// Caricamento in shared memory
    // Allocazione statica: [Y][X][Canali RGB]
	// Riduzione della Latenza: s_input risiede nella cache on-chip (L1/Shared), molto più veloce della DRAM globale
    __shared__ unsigned char s_input[SHARED_DIM][SHARED_DIM][3];
    
	// Thread linearizzati per il caricamento cooperativo
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * blockDim.y;

    for (int i = tid; i < SHARED_DIM * SHARED_DIM; i += num_threads) {
        int s_r = i / SHARED_DIM; // riga in shared
        int s_c = i % SHARED_DIM; // colonna in shared

        // Coordinate globali corrispondenti nell'immagine di Input
        int global_src_y = base_src_y + s_r;
        int global_src_x = base_src_x + s_c;

        // Clamp manuale ai bordi dell'immagine originale
        global_src_y = max(0, min(global_src_y, c_params.height - 1));
        global_src_x = max(0, min(global_src_x, c_params.width - 1));
        
		// Lettura coalesced (se possibile) e scrittura in shared
        int idx_global = (global_src_y * c_params.width + global_src_x) * c_params.channels;
        
		// Unrolling manuale dei canali per evitare loop interni nel caricamento
        if (c_params.channels == 3) {
             s_input[s_r][s_c][0] = input[idx_global + 0];
             s_input[s_r][s_c][1] = input[idx_global + 1];
             s_input[s_r][s_c][2] = input[idx_global + 2];
        }
    }
	// Barriera per attesa che tutti i thread abbiano finito caricamento cache
    __syncthreads();
	
    // Check limiti output
    if (x >= c_params.new_width || y >= c_params.new_height) return;

    // Coordinate float nello spazio input
    float gx = x * c_params.x_ratio;
    float gy = y * c_params.y_ratio;
	
    // Parte intera e frazionaria
	// Cambio variabili risp. bilineare per differenziare ma identiche realizzazioni
    int x_int = (int)gx;
    int y_int = (int)gy;
    
    float dx = gx - x_int;
    float dy = gy - y_int;
	
    // Calcolo indici relativi alla tile specifica in shared memory (mapping)
    // La shared memory inizia da (base_src_x, base_src_y) (relative coordinate)
    int s_x_int = x_int - base_src_x;
    int s_y_int = y_int - base_src_y;

    for (int c = 0; c < c_params.channels; c++) {
        float value = 0.0f;
		
        // Unrolling automatico
        #pragma unroll
        for (int m = -1; m <= 2; m++) {
			// Indice Y in shared memory: centro + offset
            int s_yy = s_y_int + m;
            // Safety check per restare dentro i limiti della shared memory allocata
            if (s_yy < 0) s_yy = 0; 
            if (s_yy >= SHARED_DIM) s_yy = SHARED_DIM - 1;

            float wy = cubic(m - dy);

            #pragma unroll
            for (int n = -1; n <= 2; n++) {
				// Indice X in shared memory: centro + offset
                int s_xx = s_x_int + n;
				
				// Safety check per restare dentro i limiti della shared memory allocata
                if (s_xx < 0) s_xx = 0;
                if (s_xx >= SHARED_DIM) s_xx = SHARED_DIM - 1;

                float wx = cubic(n - dx);
				
				// Rapida lettuara da shared memory
                float pixel = (float)s_input[s_yy][s_xx][c];
				
                value += pixel * wx * wy;
            }
        }
		// Clamp manuale
        value = fminf(fmaxf(value, 0.0f), 255.0f);
        output[(y * c_params.new_width + x) * c_params.channels + c] = (unsigned char)value;
    }
}

// Funzione host
void resize_cuda(
    unsigned char *h_input,
    unsigned char *h_output,
    int width, int height,
    int new_width, int new_height,
    int channels,
    int mode // 0=NN, 1=Bilineare, 2=Bicubica
) {
	// Alloca device memory
    unsigned char *d_input, *d_output;
    int in_size  = width * height * channels;
    int out_size = new_width * new_height * channels;
    
	// Controllo allocazione IO
    CHECK(cudaMalloc(&d_input, in_size));
    CHECK(cudaMalloc(&d_output, out_size));
    CHECK(cudaMemcpy(d_input, h_input, in_size, cudaMemcpyHostToDevice));
    
    // Configura constant memory (struct su CPU)
    KernelParams host_params;
    host_params.width = width;
    host_params.height = height;
    host_params.new_width = new_width;
    host_params.new_height = new_height;
    host_params.channels = channels;

    // Calcolo ratio qui (host) e non nel kernel per efficienza
    host_params.x_ratio = (float)width / new_width;
    host_params.y_ratio = (float)height / new_height;
    
    // Copia in constant memory con "cudaMemcpyToSymbol":
    // Copia dalla memoria host al simbolo __constant__ nel device
    CHECK(cudaMemcpyToSymbol(c_params, &host_params, sizeof(KernelParams)));

    // Imposta grid e block
    dim3 block(TILE_W, TILE_H);
    dim3 grid((new_width + block.x - 1) / block.x,
              (new_height + block.y - 1) / block.y);
    
	// Registra il tempo di inizio      
    double iStart = cpuSecond();    
    
    // lancio kernel (Nota: non si passano più width, height, ecc...)
    if (mode == 0)
        nn_kernel_const_shared<<<grid, block>>>(d_input, d_output);
    else if (mode == 1)
        bilinear_kernel_const_shared<<<grid, block>>>(d_input, d_output);
    else
        bicubic_kernel_const_shared<<<grid, block>>>(d_input, d_output);

    CHECK(cudaDeviceSynchronize());
    
	// Calcola il tempo trascorso
	double iElaps = cpuSecond() - iStart; 
	printf("kernel <<<(%d,%d), (%d,%d)>>> Time elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);	
	CHECK(cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    printf("CUDA status: %s\n", cudaGetErrorString(cudaGetLastError()));
}

// Esegue su CPU (Host)
int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s input.png scale algorithm(0 = nearest|1 = bilinear|2 = bicubic)\n", argv[0]);
        return 0;
    }
    int width, height, channels;

    // Prop. del dispositivo
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 
    printf("Nome Dispositivo: %s\n", prop.name);
    printf("Memoria Gloable Totale: %.0f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
    printf("Clock Core: %d MHz\n", prop.clockRate / 1000);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);
    
    // NomeFile.formato
    const char *imgName = argv[1];
    // Upscaling factor (moltiplicatore)
    int mul = atoi(argv[2]);
    // Tipo di interpolazione: 0 = NN, 1 = bilinear, 2 = bicubic
    int interpolation = atoi(argv[3]);

    const char *mode;
    if (interpolation == 0)
        mode = "NN";
    else if (interpolation == 1)
        mode = "BL";
    else
        mode = "BC";
	
    // Carica immagine
    unsigned char *image = stbi_load(imgName, &width, &height, &channels, 3);
    if (!image) {
        printf("Error loading image %s\n", imgName);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

	// RGB
    channels = 3;
    int new_width = width * mul;
    int new_height = height * mul;

    unsigned char *resized = (unsigned char*) malloc(new_width * new_height * channels); 
	
	// Richiama il device    
    resize_cuda(image, resized, width, height, new_width, new_height, channels, interpolation);
	
	// Salva immagine output
    char outputName[256];
    snprintf(outputName, sizeof(outputName), "cm_upscaled_x%d_%s_%s", mul, mode, imgName);
    stbi_write_png(outputName, new_width, new_height, channels, resized, new_width * channels);
    printf("\nUpscaling CUDA completato: %s\n", imgName);
    
	// Pulizia
    stbi_image_free(image);
    free(resized);
    CHECK(cudaDeviceReset());

    return 0;
}
