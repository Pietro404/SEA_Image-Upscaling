#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

//NN
__global__ void nn_shared_kernel(
    unsigned char *input,
    unsigned char *output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= new_width || y >= new_height) return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    int src_x = min((int)(x * x_ratio), width - 1);
    int src_y = min((int)(y * y_ratio), height - 1);

    int tile_x = blockIdx.x * blockDim.x;
    int tile_y = blockIdx.y * blockDim.y;

    extern __shared__ unsigned char tile[];

    int load_x = min(tile_x + tx, width - 1);
    int load_y = min(tile_y + ty, height - 1);

    for (int c = 0; c < channels; c++) {
        tile[(ty * blockDim.x + tx) * channels + c] =
            input[(load_y * width + load_x) * channels + c];
    }

    __syncthreads();

    int lx = src_x - tile_x;
    int ly = src_y - tile_y;

    lx = min(max(lx, 0), blockDim.x - 1);
    ly = min(max(ly, 0), blockDim.y - 1);

    for (int c = 0; c < channels; c++) {
        output[(y * new_width + x) * channels + c] =
            tile[(ly * blockDim.x + lx) * channels + c];
    }
}


//Bilineare
__global__ void bilinear_shared_kernel(
    unsigned char *input,
    unsigned char *output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= new_width || y >= new_height) return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float x_ratio = (float)(width - 1) / new_width;
    float y_ratio = (float)(height - 1) / new_height;

    float gx = x * x_ratio;
    float gy = y * y_ratio;

    int x0 = (int)gx;
    int y0 = (int)gy;

    float dx = gx - x0;
    float dy = gy - y0;

    int tile_x = blockIdx.x * blockDim.x;
    int tile_y = blockIdx.y * blockDim.y;

    int tile_w = blockDim.x + 1;

    extern __shared__ unsigned char tile[];

    int load_x = min(tile_x + tx, width - 1);
    int load_y = min(tile_y + ty, height - 1);

    for (int c = 0; c < channels; c++) {
        tile[(ty * tile_w + tx) * channels + c] =
            input[(load_y * width + load_x) * channels + c];
    }

    if (tx == blockDim.x - 1) {
        int hx = min(load_x + 1, width - 1);
        for (int c = 0; c < channels; c++) {
            tile[(ty * tile_w + tx + 1) * channels + c] =
                input[(load_y * width + hx) * channels + c];
        }
    }

    if (ty == blockDim.y - 1) {
        int hy = min(load_y + 1, height - 1);
        for (int c = 0; c < channels; c++) {
            tile[((ty + 1) * tile_w + tx) * channels + c] =
                input[(hy * width + load_x) * channels + c];
        }
    }

    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        int hx = min(load_x + 1, width - 1);
        int hy = min(load_y + 1, height - 1);
        for (int c = 0; c < channels; c++) {
            tile[((ty + 1) * tile_w + tx + 1) * channels + c] =
                input[(hy * width + hx) * channels + c];
        }
    }

    __syncthreads();

    int lx = x0 - tile_x;
    int ly = y0 - tile_y;

    for (int c = 0; c < channels; c++) {
        float p00 = tile[(ly * tile_w + lx) * channels + c];
        float p10 = tile[(ly * tile_w + lx + 1) * channels + c];
        float p01 = tile[((ly + 1) * tile_w + lx) * channels + c];
        float p11 = tile[((ly + 1) * tile_w + lx + 1) * channels + c];

        float value =
            p00 * (1 - dx) * (1 - dy) +
            p10 * dx * (1 - dy) +
            p01 * (1 - dx) * dy +
            p11 * dx * dy;

        output[(y * new_width + x) * channels + c] =
            (unsigned char)value;
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
__global__ void bicubic_shared_kernel(
    unsigned char *input,
    unsigned char *output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    float gx = x * x_ratio;
    float gy = y * y_ratio;

    int x_int = (int)gx;
    int y_int = (int)gy;

    float dx = gx - x_int;
    float dy = gy - y_int;

    int tile_x = blockIdx.x * blockDim.x - 1;
    int tile_y = blockIdx.y * blockDim.y - 1;

    int tile_w = blockDim.x + 3;
    int tile_h = blockDim.y + 3;

    extern __shared__ unsigned char tile[];

    // Caricamento cooperativo COMPLETO della tile
    for (int j = ty; j < tile_h; j += blockDim.y) {
        for (int i = tx; i < tile_w; i += blockDim.x) {

            int gx_i = min(max(tile_x + i, 0), width - 1);
            int gy_j = min(max(tile_y + j, 0), height - 1);

            for (int c = 0; c < channels; c++) {
                tile[(j * tile_w + i) * channels + c] =
                    input[(gy_j * width + gx_i) * channels + c];
            }
        }
    }

    __syncthreads();

    if (x >= new_width || y >= new_height) return;

    int lx = x_int - tile_x;
    int ly = y_int - tile_y;

    for (int c = 0; c < channels; c++) {
        float value = 0.0f;

        for (int m = -1; m <= 2; m++) {
            float wy = cubic(m - dy);
            for (int n = -1; n <= 2; n++) {
                float wx = cubic(n - dx);
                float pixel =
                    tile[((ly + m) * tile_w + (lx + n)) * channels + c];
                value += pixel * wx * wy;
            }
        }

        value = fminf(fmaxf(value, 0.0f), 255.0f);
        output[(y * new_width + x) * channels + c] =
            (unsigned char)value;
    }
}



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
              
    size_t sh_nn  = block.x * block.y * channels;
    size_t sh_bl  = (block.x + 1) * (block.y + 1) * channels;
    size_t sh_bc  = (block.x + 3) * (block.y + 3) * channels;          

    if (mode == 0)
        nn_shared_kernel<<<grid, block, sh_nn>>>(d_input, d_output, width, height, new_width, new_height, channels);
    else if (mode == 1)
        bilinear_shared_kernel<<<grid, block, sh_bl>>>(d_input, d_output, width, height, new_width, new_height, channels);
    else
        bicubic_shared_kernel<<<grid, block, sh_bc>>>(d_input, d_output, width, height, new_width, new_height, channels);

    cudaDeviceSynchronize();
    printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    int width, height, channels;
    
    //filename and format
    const char *imgName = "mario.jpg";
    //upscaling factor
    int mul = 8;
    //interpolation type: 0 = NN, 1 = bilinear, 2 = bicubic
    int interpolation = 1;

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

    resize_cuda(image, resized, width, height, new_width, new_height, channels, interpolation);

    stbi_write_png(outputName, new_width, new_height, channels, resized, new_width * channels);

    stbi_image_free(image);
    free(resized);

    printf("Upscaling CUDA completato\n");
    return 0;
}