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

// Macro per gestione errori CUDA
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

double cpuSecond() {
      struct timespec ts;
      timespec_get(&ts, TIME_UTC);
      return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

// --- PARAMETRI IN CONSTANT MEMORY ---
// Raggruppiamo i parametri costanti per ridurre l'uso dei registri e 
// sfruttare la cache delle costanti (velocissima per l'accesso broadcast).
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

// Costanti per Tiling
#define TILE_W 16
#define TILE_H 16
#define SHARED_DIM 32   

// ------------------------------------------------------------------
// KERNEL NEAREST NEIGHBOR (Constant + Shared)
// ------------------------------------------------------------------
__global__ void nn_kernel_const_shared(unsigned char *input, unsigned char *output) {
    // Accesso diretto a c_params senza passarli come argomenti
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calcolo base per il tile (Pre-fetching)
    int base_src_x = (int)(blockIdx.x * blockDim.x * c_params.x_ratio);
    int base_src_y = (int)(blockIdx.y * blockDim.y * c_params.y_ratio);

    __shared__ unsigned char s_input[SHARED_DIM][SHARED_DIM][3];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Caricamento in Shared Memory
    for (int i = tid; i < SHARED_DIM * SHARED_DIM; i += (blockDim.x * blockDim.y)) {
        int s_r = i / SHARED_DIM;
        int s_c = i % SHARED_DIM;
        int gx = min(base_src_x + s_c, c_params.width - 1);
        int gy = min(base_src_y + s_r, c_params.height - 1);
        
        int g_idx = (gy * c_params.width + gx) * c_params.channels;
        
        if (c_params.channels == 3) {
            s_input[s_r][s_c][0] = input[g_idx + 0];
            s_input[s_r][s_c][1] = input[g_idx + 1];
            s_input[s_r][s_c][2] = input[g_idx + 2];
        }
    }
    __syncthreads();

    if (x >= c_params.new_width || y >= c_params.new_height) return;

    int src_x = (int)(x * c_params.x_ratio);
    int src_y = (int)(y * c_params.y_ratio);

    int s_x = src_x - base_src_x;
    int s_y = src_y - base_src_y;
    
    s_x = max(0, min(s_x, SHARED_DIM - 1));
    s_y = max(0, min(s_y, SHARED_DIM - 1));

    for (int c = 0; c < c_params.channels; c++) {
        output[(y * c_params.new_width + x) * c_params.channels + c] = s_input[s_y][s_x][c];
    }
}

// ------------------------------------------------------------------
// KERNEL BILINEARE (Constant + Shared)
// ------------------------------------------------------------------
__global__ void bilinear_kernel_const_shared(unsigned char *input, unsigned char *output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int base_src_x = (int)((blockIdx.x * blockDim.x) * c_params.x_ratio);
    int base_src_y = (int)((blockIdx.y * blockDim.y) * c_params.y_ratio);

    __shared__ unsigned char s_input[SHARED_DIM][SHARED_DIM][3];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < SHARED_DIM * SHARED_DIM; i += (blockDim.x * blockDim.y)) {
        int s_r = i / SHARED_DIM;
        int s_c = i % SHARED_DIM;
        int gx = min(base_src_x + s_c, c_params.width - 1);
        int gy = min(base_src_y + s_r, c_params.height - 1);
        int g_idx = (gy * c_params.width + gx) * c_params.channels;
        
        if (c_params.channels == 3) {
            s_input[s_r][s_c][0] = input[g_idx + 0];
            s_input[s_r][s_c][1] = input[g_idx + 1];
            s_input[s_r][s_c][2] = input[g_idx + 2];
        }
    }
    __syncthreads();

    if (x >= c_params.new_width || y >= c_params.new_height) return;

    float gx = x * c_params.x_ratio;
    float gy = y * c_params.y_ratio;
    int x0 = (int)gx;
    int y0 = (int)gy;
    float dx = gx - x0;
    float dy = gy - y0;

    int s_x0 = x0 - base_src_x;
    int s_y0 = y0 - base_src_y;

    for (int c = 0; c < c_params.channels; c++) {
        // Clamp manuale per sicurezza sui bordi del tile
        int sx0_cl = max(0, min(s_x0, SHARED_DIM - 2)); 
        int sy0_cl = max(0, min(s_y0, SHARED_DIM - 2));

        float p00 = s_input[sy0_cl][sx0_cl][c];
        float p10 = s_input[sy0_cl][sx0_cl + 1][c];
        float p01 = s_input[sy0_cl + 1][sx0_cl][c];
        float p11 = s_input[sy0_cl + 1][sx0_cl + 1][c];

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

// ------------------------------------------------------------------
// KERNEL BICUBICA (Constant + Shared)
// ------------------------------------------------------------------
__global__ void bicubic_kernel_const_shared(unsigned char *input, unsigned char *output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int start_out_x = blockIdx.x * blockDim.x;
    int start_out_y = blockIdx.y * blockDim.y;

    int base_src_x = (int)(start_out_x * c_params.x_ratio);
    int base_src_y = (int)(start_out_y * c_params.y_ratio);

    // Halo correction
    base_src_x -= 1;
    base_src_y -= 1;

    __shared__ unsigned char s_input[SHARED_DIM][SHARED_DIM][3];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < SHARED_DIM * SHARED_DIM; i += (blockDim.x * blockDim.y)) {
        int s_r = i / SHARED_DIM;
        int s_c = i % SHARED_DIM;

        int global_src_y = base_src_y + s_r;
        int global_src_x = base_src_x + s_c;

        global_src_y = max(0, min(global_src_y, c_params.height - 1));
        global_src_x = max(0, min(global_src_x, c_params.width - 1));

        int idx_global = (global_src_y * c_params.width + global_src_x) * c_params.channels;
        
        if (c_params.channels == 3) {
             s_input[s_r][s_c][0] = input[idx_global + 0];
             s_input[s_r][s_c][1] = input[idx_global + 1];
             s_input[s_r][s_c][2] = input[idx_global + 2];
        }
    }
    __syncthreads();

    if (x >= c_params.new_width || y >= c_params.new_height) return;

    float gx = x * c_params.x_ratio;
    float gy = y * c_params.y_ratio;

    int x_int = (int)gx;
    int y_int = (int)gy;
    
    float dx = gx - x_int;
    float dy = gy - y_int;

    int s_x_int = x_int - base_src_x;
    int s_y_int = y_int - base_src_y;

    for (int c = 0; c < c_params.channels; c++) {
        float value = 0.0f;

        #pragma unroll
        for (int m = -1; m <= 2; m++) {
            int s_yy = s_y_int + m;
            // Clamp safe
            if (s_yy < 0) s_yy = 0; 
            if (s_yy >= SHARED_DIM) s_yy = SHARED_DIM - 1;

            float wy = cubic(m - dy);

            #pragma unroll
            for (int n = -1; n <= 2; n++) {
                int s_xx = s_x_int + n;
                if (s_xx < 0) s_xx = 0;
                if (s_xx >= SHARED_DIM) s_xx = SHARED_DIM - 1;

                float wx = cubic(n - dx);
                float pixel = (float)s_input[s_yy][s_xx][c];
                value += pixel * wx * wy;
            }
        }
        value = fminf(fmaxf(value, 0.0f), 255.0f);
        output[(y * c_params.new_width + x) * c_params.channels + c] = (unsigned char)value;
    }
}

// ------------------------------------------------------------------
// HOST FUNCTION
// ------------------------------------------------------------------
void resize_cuda(
    unsigned char *h_input,
    unsigned char *h_output,
    int width, int height,
    int new_width, int new_height,
    int channels,
    int mode // 0=NN, 1=Bilineare, 2=Bicubica
) {
    unsigned char *d_input, *d_output;
    int in_size  = width * height * channels;
    int out_size = new_width * new_height * channels;
    
    CHECK(cudaMalloc(&d_input, in_size));
    CHECK(cudaMalloc(&d_output, out_size));
    CHECK(cudaMemcpy(d_input, h_input, in_size, cudaMemcpyHostToDevice));
    
    // --- SETUP CONSTANT MEMORY ---
    // Prepariamo la struct sulla CPU
    KernelParams host_params;
    host_params.width = width;
    host_params.height = height;
    host_params.new_width = new_width;
    host_params.new_height = new_height;
    host_params.channels = channels;

    // Calcoliamo i ratio qui (Host) e non nel kernel per efficienza
    host_params.x_ratio = (float)width / new_width;
    host_params.y_ratio = (float)height / new_height;
    
    // COPIA IN CONSTANT MEMORY
    // cudaMemcpyToSymbol copia dalla memoria host al simbolo __constant__ nel device
    CHECK(cudaMemcpyToSymbol(c_params, &host_params, sizeof(KernelParams)));

    // Setup Grid
    dim3 block(TILE_W, TILE_H);
    dim3 grid((new_width + block.x - 1) / block.x,
              (new_height + block.y - 1) / block.y);
              
    double iStart = cpuSecond();    
    
    // Lancio Kernel (Notare che non passiamo più width, height, etc.)
    if (mode == 0)
        nn_kernel_const_shared<<<grid, block>>>(d_input, d_output);
    else if (mode == 1)
        bilinear_kernel_const_shared<<<grid, block>>>(d_input, d_output);
    else
        bicubic_kernel_const_shared<<<grid, block>>>(d_input, d_output);

    CHECK(cudaDeviceSynchronize());
    
    double iElaps = cpuSecond() - iStart;
    
    // Fix per il warning printf (accediamo a .x e .y)
    printf("kernel <<<(%d,%d), (%d,%d)>>> Time elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);    
    
    CHECK(cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    printf("CUDA status: %s\n", cudaGetErrorString(cudaGetLastError()));
}

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

    unsigned char *image = stbi_load(imgName, &width, &height, &channels, 3);
    if (!image) {
        printf("Error loading image %s\n", imgName);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

    channels = 3;
    int new_width = width * mul;
    int new_height = height * mul;

    unsigned char *resized = (unsigned char*) malloc(new_width * new_height * channels); 
    
    resize_cuda(image, resized, width, height, new_width, new_height, channels, interpolation);
    
    char outputName[256];
    snprintf(outputName, sizeof(outputName), "cm_upscaled_x%d_%s_%s", mul, modeStr, imgName);
    stbi_write_png(outputName, new_width, new_height, channels, resized, new_width * channels);
    printf("\nUpscaling CUDA completato: %s\n", outputName);
    
    stbi_image_free(image);
    free(resized);
    CHECK(cudaDeviceReset());

    return 0;

}
