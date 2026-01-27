// main.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHANNELS 3
#define TILE_SIZE 16

// ============================
// Utility CUDA
// ============================

__device__ float cubic(float x) {
    x = fabsf(x);
    if (x <= 1.0f)
        return (1.5f * x - 2.5f) * x * x + 1.0f;
    else if (x < 2.0f)
        return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
    return 0.0f;
}

__device__ int clamp(int v, int low, int high) {
    return max(low, min(v, high));
}

// ============================
// Kernel 1: Nearest Neighbor
// ============================

__global__ void upscale_nearest(
    unsigned char* input,
    unsigned char* output,
    int inW, int inH,
    int outW, int outH,
    float scale
) {
    __shared__ unsigned char tile[(TILE_SIZE + 1) * (TILE_SIZE + 1) * CHANNELS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    if (x >= outW || y >= outH) return;

    float srcX = x / scale;
    float srcY = y / scale;

    int ix = clamp((int)(srcX + 0.5f), 0, inW - 1);
    int iy = clamp((int)(srcY + 0.5f), 0, inH - 1);

    int inIdx = (iy * inW + ix) * CHANNELS;
    int outIdx = (y * outW + x) * CHANNELS;

    for (int c = 0; c < CHANNELS; c++)
        output[outIdx + c] = input[inIdx + c];
}

// ============================
// Kernel 2: Bilinear
// ============================

__global__ void upscale_bilinear(
    unsigned char* input,
    unsigned char* output,
    int inW, int inH,
    int outW, int outH,
    float scale
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outW || y >= outH) return;

    float gx = x / scale;
    float gy = y / scale;

    int x0 = clamp((int)floorf(gx), 0, inW - 1);
    int y0 = clamp((int)floorf(gy), 0, inH - 1);
    int x1 = clamp(x0 + 1, 0, inW - 1);
    int y1 = clamp(y0 + 1, 0, inH - 1);

    float dx = gx - x0;
    float dy = gy - y0;

    int idx00 = (y0 * inW + x0) * CHANNELS;
    int idx10 = (y0 * inW + x1) * CHANNELS;
    int idx01 = (y1 * inW + x0) * CHANNELS;
    int idx11 = (y1 * inW + x1) * CHANNELS;

    int outIdx = (y * outW + x) * CHANNELS;

    for (int c = 0; c < CHANNELS; c++) {
        float v00 = input[idx00 + c];
        float v10 = input[idx10 + c];
        float v01 = input[idx01 + c];
        float v11 = input[idx11 + c];

        float v0 = v00 + dx * (v10 - v00);
        float v1 = v01 + dx * (v11 - v01);
        float v = v0 + dy * (v1 - v0);

        output[outIdx + c] = (unsigned char)clamp((int)(v + 0.5f), 0, 255);
    }
}

// ============================
// Kernel 3: Bicubic
// ============================

__global__ void upscale_bicubic(
    unsigned char* input,
    unsigned char* output,
    int inW, int inH,
    int outW, int outH,
    float scale
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outW || y >= outH) return;

    float gx = x / scale;
    float gy = y / scale;

    int ix = (int)floorf(gx);
    int iy = (int)floorf(gy);

    float dx = gx - ix;
    float dy = gy - iy;

    int outIdx = (y * outW + x) * CHANNELS;

    for (int c = 0; c < CHANNELS; c++) {
        float sum = 0.0f;
        float wsum = 0.0f;

        for (int m = -1; m <= 2; m++) {
            for (int n = -1; n <= 2; n++) {
                int sx = clamp(ix + n, 0, inW - 1);
                int sy = clamp(iy + m, 0, inH - 1);

                float wx = cubic(n - dx);
                float wy = cubic(m - dy);
                float w = wx * wy;

                int idx = (sy * inW + sx) * CHANNELS;
                sum += w * input[idx + c];
                wsum += w;
            }
        }

        float v = sum / wsum;
        output[outIdx + c] = (unsigned char)clamp((int)(v + 0.5f), 0, 255);
    }
}

// ============================
// MAIN
// ============================

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s input.png output.png scale algorithm(nearest|bilinear|bicubic)\n", argv[0]);
        return 0;
    }

    const char* inputPath = argv[1];
    const char* outputPath = argv[2];
    float scale = atof(argv[3]);
    const char* algo = argv[4];

    int w, h, ch;
    unsigned char* img = stbi_load(inputPath, &w, &h, &ch, CHANNELS);
    if (!img) {
        printf("Error loading image\n");
        return -1;
    }

    int outW = (int)(w * scale);
    int outH = (int)(h * scale);
    size_t inSize = w * h * CHANNELS;
    size_t outSize = outW * outH * CHANNELS;

    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, inSize);
    cudaMalloc(&d_out, outSize);

    cudaMemcpy(d_in, img, inSize, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);

    if (strcmp(algo, "nearest") == 0) {
        upscale_nearest<<<grid, block>>>(d_in, d_out, w, h, outW, outH, scale);
    } 
    else if (strcmp(algo, "bilinear") == 0) {
        upscale_bilinear<<<grid, block>>>(d_in, d_out, w, h, outW, outH, scale);
    } 
    else if (strcmp(algo, "bicubic") == 0) {
        upscale_bicubic<<<grid, block>>>(d_in, d_out, w, h, outW, outH, scale);
    } 
    else {
        printf("Unknown algorithm\n");
        return -1;
    }

    cudaDeviceSynchronize();

    unsigned char* outImg = (unsigned char*)malloc(outSize);
    cudaMemcpy(outImg, d_out, outSize, cudaMemcpyDeviceToHost);

    stbi_write_png(outputPath, outW, outH, CHANNELS, outImg, outW * CHANNELS);

    stbi_image_free(img);
    free(outImg);
    cudaFree(d_in);
    cudaFree(d_out);

    printf("Upscaling completed: %s\n", outputPath);
    return 0;
}
