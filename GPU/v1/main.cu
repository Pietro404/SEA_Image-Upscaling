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

//NN
__global__ void nn_kernel(
    unsigned char *input,
    unsigned char *output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Controllo dei bordi: assicura che il thread sia dentro l'immagine 
    if (x >= new_width || y >= new_height) return;
	
	//TODO: eseguire test solo int eliminando da qui
    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;
	
    int src_x = (int)(x * x_ratio);
    int src_y = (int)(y * y_ratio);
	//a qui

	/*e mantenere solo questo
	int src_x = (x * width) / new_width;
    int src_y = (y * height) / new_height;
	*/

    //vedere se tenere
    if (src_x >= width)  src_x = width - 1;
    if (src_y >= height) src_y = height - 1;
	
	//Calcola l'indice di base: baseIndex = (i * width + j) * 3
	//Accesso ai canali: R=baseI, G=baseI+1, B=baseI+2
    for (int c = 0; c < channels; c++) {
        output[(y * new_width + x) * channels + c] =
            input[(src_y * width + src_x) * channels + c];
    }
    
    /*nessuna variazione significativa come in cpu
    int in_idx = (src_y * width + src_x) * channels;
    int out_idx = (y * new_width + x) * channels;

    output[out_idx + 0] = input[in_idx + 0];
    output[out_idx + 1] = input[in_idx + 1];
    output[out_idx + 2] = input[in_idx + 2];
    */
}

//Bilineare
__global__ void bilinear_kernel(
    unsigned char *input,
    unsigned char *output,
    int width, int height,
    int new_width, int new_height,
    int channels
	//float x_ratio, float y_ratio
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Controllo dei bordi: assicura che il thread sia dentro l'immagine 
    if (x >= new_width || y >= new_height) return;

	//TODO: PROVARE a passare valore fix da CPU!
	//Eliminare x/y_ratio qua e aggiungere lato host
    float x_ratio = (float)(width - 1) / new_width;
    float y_ratio = (float)(height - 1) / new_height;

    float gx = x * x_ratio;
    float gy = y * y_ratio;

    int x0 = (int)gx;
    int y0 = (int)gy;
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);

    float dx = gx - x0;
    float dy = gy - y0;

    for (int c = 0; c < channels; c++) {
        float p00 = input[(y0 * width + x0) * channels + c];
        float p10 = input[(y0 * width + x1) * channels + c];
        float p01 = input[(y1 * width + x0) * channels + c];
        float p11 = input[(y1 * width + x1) * channels + c];

        float value =
            p00 * (1 - dx) * (1 - dy) +
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
__global__ void bicubic_kernel(
    unsigned char *input,
    unsigned char *output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Controllo dei bordi: assicura che il thread sia dentro l'immagine 
    if (x >= new_width || y >= new_height) return;

    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    float gx = x * x_ratio;
    float gy = y * y_ratio;

    int x_int = (int)gx;
    int y_int = (int)gy;

    float dx = gx - x_int;
    float dy = gy - y_int;

    for (int c = 0; c < channels; c++) {
        float value = 0.0f;

        for (int m = -1; m <= 2; m++) {
            int yy = min(max(y_int + m, 0), height - 1);
            float wy = cubic(m - dy);

            for (int n = -1; n <= 2; n++) {
                int xx = min(max(x_int + n, 0), width - 1);
                float wx = cubic(n - dx);

                float pixel = input[(yy * width + xx) * channels + c];
                value += pixel * wx * wy;
            }
        }

        value = fminf(fmaxf(value, 0.0f), 255.0f);
        output[(y * new_width + x) * channels + c] = (unsigned char)value;
    }
}

//esegue lato GPU
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
    
    //2D Set up grid and block dimensions 16X16=256t
    dim3 block(16, 16);
    dim3 grid((new_width + block.x - 1) / block.x,
              (new_height + block.y - 1) / block.y);
			  
	// Registra il tempo di inizio
	double iStart = cpuSecond();	
	
    //Configurazione di esecuzione: nGrid blocco, nBlock thread. Avvia nBlock istanze parallele del kernel sulla GPU
    if (mode == 0)
        nn_kernel<<<grid, block>>>(d_input, d_output, width, height, new_width, new_height, channels);
    else if (mode == 1)
        bilinear_kernel<<<grid, block>>>(d_input, d_output, width, height, new_width, new_height, channels);
    else
        bicubic_kernel<<<grid, block>>>(d_input, d_output, width, height, new_width, new_height, channels);

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
	//provare a passare float x_ratio, float y_ratio direttamente da qua
	//float x_ratio = (float)(width - 1) / new_width;
    //float y_ratio = (float)(height - 1) / new_height;
	//riscrivere resize_cuda con param()
    resize_cuda(image, resized, width, height, new_width, new_height, channels, interpolation);
    
    // Save the output image
    char outputName[256];
    snprintf(outputName, sizeof(outputName), "upscaled_x%d_%s_%s", mul, mode, imgName);
    stbi_write_png(outputName, new_width, new_height, channels, resized, new_width * channels);
    printf("\nUpscaling CUDA di %s completato\n", imgName);
    
    // Clean up
    stbi_image_free(image);
    free(resized);
    CHECK(cudaDeviceReset());

    return 0;
}







