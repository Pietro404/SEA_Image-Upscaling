#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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

    if (x >= new_width || y >= new_height) return;

    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    int src_x = (int)(x * x_ratio);
    int src_y = (int)(y * y_ratio);

    if (src_x >= width)  src_x = width - 1;
    if (src_y >= height) src_y = height - 1;

    for (int c = 0; c < channels; c++) {
        output[(y * new_width + x) * channels + c] =
            input[(src_y * width + src_x) * channels + c];
    }
}

//Bilineare
__global__ void bilinear_kernel(
    unsigned char *input,
    unsigned char *output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= new_width || y >= new_height) return;

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
    unsigned char *d_input, *d_output;

    int in_size  = width * height * channels;
    int out_size = new_width * new_height * channels;

    cudaMalloc(&d_input, in_size);
    cudaMalloc(&d_output, out_size);

    cudaMemcpy(d_input, h_input, in_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((new_width + block.x - 1) / block.x,
              (new_height + block.y - 1) / block.y);
    //Configurazione di esecuzione: nGrid blocco, nBlock thread. Avvia nBlock istanze parallele del kernel sulla GPU
    if (mode == 0)
        nn_kernel<<<grid, block>>>(d_input, d_output, width, height, new_width, new_height, channels);
    else if (mode == 1)
        bilinear_kernel<<<grid, block>>>(d_input, d_output, width, height, new_width, new_height, channels);
    else
        bicubic_kernel<<<grid, block>>>(d_input, d_output, width, height, new_width, new_height, channels);

    cudaDeviceSynchronize();
    printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

//esegue su CPU
int main() {
    int width, height, channels;

    //---chk--->Proprietà del dispositivo
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 
    printf("Nome Dispositivo: %s\n", prop.name);
    printf("Memoria Gloable Totale: %.0f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
    printf("Clock Core: %d MHz\n", prop.clockRate / 1000);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    //---
    
    //filename and format
    const char *imgName = "mario.jpg";
    //upscaling factor
    int mul = 2;
    //interpolation type: 0 = NN, 1 = bilinear, 2 = bicubic
    int interpolation = 2;

    const char *mode;
    if (interpolation == 0)
        mode = "NN";
    else if (interpolation == 1)
        mode = "BL";
    else
        mode = "BC";

    char outputName[256];
    snprintf(outputName, sizeof(outputName), "upscaled_x%d_%s_%s", mul, mode, imgName);

    unsigned char *image = stbi_load(imgName, &width, &height, &channels, 3);
    if (!image) {
        printf("Errore caricamento immagine\n");
        return 1;
    }

    channels = 3;
    int new_width = width * mul;
    int new_height = height * mul;

    unsigned char *resized = (unsigned char*) malloc(new_width * new_height * channels);
    
    //chiama device
    resize_cuda(image, resized, width, height, new_width, new_height, channels, interpolation);

    stbi_write_png(outputName, new_width, new_height, channels, resized, new_width * channels);

    stbi_image_free(image);
    free(resized);

    printf("Upscaling CUDA completato\n");
    return 0;
}
