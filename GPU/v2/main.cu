#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <string.h>

// Include STB image libraries
//rimuove warnings
#pragma nv_diag_suppress 550
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//Fornisce file, riga, codice e descrizione dell'errore
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
//Funzione timer CPU: misura il tempo wall-clock visto dalla CPU
//Imprecisa, genera overhead, deprecata
double cpuSecond() {
      struct timespec ts;
      timespec_get(&ts, TIME_UTC);
      return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
    }

// Dimensione blocco di thread (Output): 16x16=256t
#define TILE_W 16
#define TILE_H 16

// Dimensione cache in Shared Memory (Input)
// Deve essere sufficiente a contenere i pixel di input necessari per un blocco 16x16
// In caso di Upscaling, l'input è più piccolo dell'output, ma serve aggiungere margine per l'Halo
// base: 32x32 dimensione sicura per coprire l'input + halo quando scale >= 1
// per downscaling maggiore: test aumentare dimensione 
#define SHARED_DIM 32    

//NN
__global__ void nn_kernel_shared(
    unsigned char *input, unsigned char *output,
    int width, int height, int new_width, int new_height, int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    // Coordinate di base per il caricamento della tile
    int base_src_x = (int)(blockIdx.x * blockDim.x * x_ratio);
    int base_src_y = (int)(blockIdx.y * blockDim.y * y_ratio);

	//Riduzione della Latenza: s_input risiede nella cache on-chip (L1/Shared), molto più veloce della DRAM globale
    __shared__ unsigned char s_input[SHARED_DIM][SHARED_DIM][3];

    // Caricamento cooperativo
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < SHARED_DIM * SHARED_DIM; i += (blockDim.x * blockDim.y)) {
        int s_r = i / SHARED_DIM;
        int s_c = i % SHARED_DIM;
        int gx = min(base_src_x + s_c, width - 1);
        int gy = min(base_src_y + s_r, height - 1);
        
        int g_idx = (gy * width + gx) * channels;
        s_input[s_r][s_c][0] = input[g_idx + 0];
        s_input[s_r][s_c][1] = input[g_idx + 1];
        s_input[s_r][s_c][2] = input[g_idx + 2];
    }
    __syncthreads();

    if (x >= new_width || y >= new_height) return;

    // Logica NN: prendi il pixel più vicino
    int src_x = (int)(x * x_ratio);
    int src_y = (int)(y * y_ratio);

    // Mapping su shared memory
    int s_x = src_x - base_src_x;
    int s_y = src_y - base_src_y;
    
    // Clamp per sicurezza (evita fuori intervallo shared)
    s_x = max(0, min(s_x, SHARED_DIM - 1));
    s_y = max(0, min(s_y, SHARED_DIM - 1));

    for (int c = 0; c < channels; c++) {
        output[(y * new_width + x) * channels + c] = s_input[s_y][s_x][c];
    }
}

//Bilineare
__global__ void bilinear_kernel_shared(
    unsigned char *input, unsigned char *output,
    int width, int height, int new_width, int new_height, int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    // L'angolo in alto a sinistra della zona di input necessaria al blocco
    int base_src_x = (int)((blockIdx.x * blockDim.x) * x_ratio);
    int base_src_y = (int)((blockIdx.y * blockDim.y) * y_ratio);
	
	//Riduzione della Latenza: s_input risiede nella cache on-chip (L1/Shared), molto più veloce della DRAM globale
    __shared__ unsigned char s_input[SHARED_DIM][SHARED_DIM][3];

    // Caricamento cooperativo (identico alla bicubica, copre l'area SHARED_DIM x SHARED_DIM)
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < SHARED_DIM * SHARED_DIM; i += (blockDim.x * blockDim.y)) {
        int s_r = i / SHARED_DIM;
        int s_c = i % SHARED_DIM;
        int gx = min(base_src_x + s_c, width - 1);
        int gy = min(base_src_y + s_r, height - 1);
        int g_idx = (gy * width + gx) * channels;
        s_input[s_r][s_c][0] = input[g_idx + 0];
        s_input[s_r][s_c][1] = input[g_idx + 1];
        s_input[s_r][s_c][2] = input[g_idx + 2];
    }
    __syncthreads();

    if (x >= new_width || y >= new_height) return;

    float gx = x * x_ratio;
    float gy = y * y_ratio;
    int x0 = (int)gx;
    int y0 = (int)gy;
    float dx = gx - x0;
    float dy = gy - y0;

    // Coordinate relative alla shared memory
    int s_x0 = x0 - base_src_x;
    int s_y0 = y0 - base_src_y;

    for (int c = 0; c < channels; c++) {
        // Lettura dei 4 vicini dalla Shared Memory
        float p00 = s_input[s_y0][s_x0][c];
        float p10 = s_input[s_y0][s_x0 + 1][c];
        float p01 = s_input[s_y0 + 1][s_x0][c];
        float p11 = s_input[s_y0 + 1][s_x0 + 1][c];

        float value = p00 * (1 - dx) * (1 - dy) +
                      p10 * dx * (1 - dy) +
                      p01 * (1 - dx) * dy +
                      p11 * dx * dy;

        output[(y * new_width + x) * channels + c] = (unsigned char)value;
    }
}

__device__ float cubic(float x) {
    const float a = -0.5f;
    x = fabsf(x);

    if (x <= 1.0f)
        return (a + 2) * x * x * x - (a + 3) * x * x + 1;
    else if (x < 2.0f)
        return a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a;
    else
        return 0.0f;
}

//Bicubica
__global__ void bicubic_kernel_shared(
    unsigned char *input,
    unsigned char *output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    // Coordinate globali del pixel di OUTPUT
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Rapporto di scala
    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    // #1 - IDENTIFICAZIONE AREA INPUT ROI (Region of Interest)
    // calcolo di quale porzione dell'immagine di input serve a QUESTO blocco di thread
    // trovare il pixel input corrispondente all'angolo in alto a sinistra del blocco output
    int start_out_x = blockIdx.x * blockDim.x;
    int start_out_y = blockIdx.y * blockDim.y;

    // angolo del blocco output mappato sullo spazio input
    int base_src_x = (int)(start_out_x * x_ratio);
    int base_src_y = (int)(start_out_y * y_ratio);

    // origine indietro di 1 per coprire l'Halo sinistro/superiore (necessario per bicubica m=-1)
    base_src_x -= 1;
    base_src_y -= 1;

    // #2 - CARICAMENTO IN SHARED MEMORY
    // Allocazione statica: [Y][X][Canali RGB]
    __shared__ unsigned char s_input[SHARED_DIM][SHARED_DIM][3];

    // thread linearizzati per il caricamento cooperativo
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;

    // Ogni thread carica uno o più pixel nella shared memory finché non riempiamo SHARED_DIM x SHARED_DIM
    // Nota: In upscaling, l'area utile reale è piccola, ma carichiamo una regione fissa per semplicità e sicurezza.
    for (int i = tid; i < SHARED_DIM * SHARED_DIM; i += num_threads) {
        int s_r = i / SHARED_DIM; // Riga in shared
        int s_c = i % SHARED_DIM; // Colonna in shared

        // Coordinate globali corrispondenti nell'immagine di Input
        int global_src_y = base_src_y + s_r;
        int global_src_x = base_src_x + s_c;

        // Clamp ai bordi dell'immagine originale
        global_src_y = max(0, min(global_src_y, height - 1));
        global_src_x = max(0, min(global_src_x, width - 1));

        // Lettura coalesced (se possibile) e scrittura in shared
        int idx_global = (global_src_y * width + global_src_x) * channels;
        
        // Unrolling manuale dei canali per evitare loop interni nel caricamento
        if (channels == 3) {
             s_input[s_r][s_c][0] = input[idx_global + 0];
             s_input[s_r][s_c][1] = input[idx_global + 1];
             s_input[s_r][s_c][2] = input[idx_global + 2];
        }
    }

    // Barriera: aspettiamo che tutti i thread abbiano finito di caricare la cache
    __syncthreads();

    // --- 3. CALCOLO BICUBICO (Leggendo SOLO da Shared Memory) ---
    
    // Check limiti output
    if (x >= new_width || y >= new_height) return;

    // Coordinate float nell'input space
    float gx = x * x_ratio;
    float gy = y * y_ratio;

    // Parte intera e frazionaria
    int x_int = (int)gx;
    int y_int = (int)gy;
    
    float dx = gx - x_int;
    float dy = gy - y_int;

    // Calcoliamo gli indici relativi alla nostra tile in Shared Memory
    // La nostra shared memory inizia da (base_src_x, base_src_y)
    int s_x_int = x_int - base_src_x;
    int s_y_int = y_int - base_src_y;

    for (int c = 0; c < channels; c++) {
        float value = 0.0f;

        // Kernel Bicubico 4x4
        #pragma unroll
        for (int m = -1; m <= 2; m++) {
            // Indice Y in shared memory: il centro + offset
            int s_yy = s_y_int + m;
            
            // Safety check: restiamo dentro i limiti della shared memory allocata
            // (Con SHARED_DIM=32 e un blocco 16x16 in upscaling, non dovremmo mai uscire)
            if (s_yy < 0) s_yy = 0; 
            if (s_yy >= SHARED_DIM) s_yy = SHARED_DIM - 1;

            float wy = cubic(m - dy);

            #pragma unroll
            for (int n = -1; n <= 2; n++) {
                // Indice X in shared memory
                int s_xx = s_x_int + n;
                
                if (s_xx < 0) s_xx = 0;
                if (s_xx >= SHARED_DIM) s_xx = SHARED_DIM - 1;

                float wx = cubic(n - dx);

                // LETTURA DALLA SHARED MEMORY (Veloce!)
                float pixel = (float)s_input[s_yy][s_xx][c];
                
                value += pixel * wx * wy;
            }
        }

        value = fminf(fmaxf(value, 0.0f), 255.0f);
        output[(y * new_width + x) * channels + c] = (unsigned char)value;
    }
}

//host function
void resize_cuda(
    unsigned char *h_input,
    unsigned char *h_output,
    int width, int height,
    int new_width, int new_height,
    int channels,
    int mode // 0=NN, 1=Bilineare, 2=Bicubica
) {
    // Allocate device memory
    unsigned char *d_input, *d_output;
    int in_size  = width * height * channels;
    int out_size = new_width * new_height * channels;
    
    //controllo allocazione IO
    CHECK(cudaMalloc(&d_input, in_size));
    CHECK(cudaMalloc(&d_output, out_size));
    CHECK(cudaMemcpy(d_input, h_input, in_size, cudaMemcpyHostToDevice));
    
    // Set up grid and block dimensions
    dim3 block(TILE_W, TILE_H);
    dim3 grid((new_width + block.x - 1) / block.x,
              (new_height + block.y - 1) / block.y);
			  
	// Registra il tempo di inizio
	double iStart = cpuSecond();	
	
    //Configurazione di esecuzione: nGrid blocco, nBlock thread. Avvia nBlock istanze parallele del kernel sulla GPU
    if (mode == 0)
        nn_kernel_shared<<<grid, block>>>(d_input, d_output, width, height, new_width, new_height, channels);
    else if (mode == 1)
        bilinear_kernel_shared<<<grid, block>>>(d_input, d_output, width, height, new_width, new_height, channels);
    else
        bicubic_kernel_shared<<<grid, block>>>(d_input, d_output, width, height, new_width, new_height, channels);

    CHECK(cudaDeviceSynchronize());
	
	// Calcola il tempo trascorso
	double iElaps = cpuSecond() - iStart;
  //risolve warning dim3 type 
  printf("kernel <<<(%d,%d), (%d,%d)>>> Time elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);	
	CHECK(cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
}

//esegue su CPU
int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s input.png scale algorithm(0 = nearest|1 = bilinear|2 = bicubic)\n", argv[0]);
        return 0;
    }
    int width, height, channels;

    //---chk--->Prop. del dispositivo
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 
    printf("Nome Dispositivo: %s\n", prop.name);
    printf("Memoria Gloable Totale: %.0f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
    printf("Clock Core: %d MHz\n", prop.clockRate / 1000);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);
    //---
    
    //filename and format
    const char *imgName = argv[1];
    //upscaling factor
    int mul = atoi(argv[2]);
    //interpolation type: 0 = NN, 1 = bilinear, 2 = bicubic
    int interpolation = atoi(argv[3]);

    const char *mode;
    if (interpolation == 0)
        mode = "NN";
    else if (interpolation == 1)
        mode = "BL";
    else
        mode = "BC";

    //load image
    unsigned char *image = stbi_load(imgName, &width, &height, &channels, 3);
    if (!image) {
        printf("Error loading image %s\n", imgName);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

	//RGB
    channels = 3;
    int new_width = width * mul;
    int new_height = height * mul;

    unsigned char *resized = (unsigned char*) malloc(new_width * new_height * channels); 
    
	//calls device
    resize_cuda(image, resized, width, height, new_width, new_height, channels, interpolation);
    
    // Save the output image
    char outputName[256];
    snprintf(outputName, sizeof(outputName), "sm_upscaled_x%d_%s_%s", mul, mode, imgName);
    stbi_write_png(outputName, new_width, new_height, channels, resized, new_width * channels);
    printf("\nUpscaling CUDA di %s completato\n", imgName);
    
    // Clean up
    stbi_image_free(image);
    free(resized);
    CHECK(cudaDeviceReset());

    return 0;
}








