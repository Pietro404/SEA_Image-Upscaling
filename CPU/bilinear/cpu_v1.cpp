#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "test.h"


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <vector>

// Bilinear RGB 
void cpu_bil_v1(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    for (int j = 0; j < new_height; j++) {
        float gy = j * y_ratio;
        int y1 = (int)gy;
        float dy = gy - y1;
        int y2 = (y1 + 1 < height - 1) ? y1 + 1 : height - 1; // min

        for (int i = 0; i < new_width; i++) {
            float gx = i * x_ratio;
            int x1 = (int)gx;
            float dx = gx - x1;
            int x2 = (x1 + 1 < width - 1) ? x1 + 1  : width -1; // min

            for (int c = 0; c < channels; c++) {
                unsigned char p00 = input[(y1  * width + x1 ) * channels + c];
                unsigned char p10 = input[(y1  * width + x2) * channels + c];
                unsigned char p01 = input[(y2 * width + x1 ) * channels + c];
                unsigned char p11 = input[(y2 * width + x2) * channels + c];

                //F(x,y)=(1−a)(1−b)F(i,j) + a(1−b)F(i+1,j) + (1−a)bF(i,j+1) + abF(i+1,j+1)
                float value =
                    p00 * (1 - dx) * (1 - dy) +
                    p10 * dx       * (1 - dy) +
                    p01 * (1 - dx) * dy       +
                    p11 * dx       * dy;

                output[(j * new_width + i) * channels + c] =
                    (unsigned char)(value);
            }
        }
    }
}

// Ottimizzazione con fixed-point e LUT per X  Q16.16 (scalare)
void cpu_bil_v2(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    const int FP_SHIFT = 16;
    const int FP_ONE   = 1 << FP_SHIFT;

    // Rapporti di scala
    int x_ratio = ((width ) << FP_SHIFT) / new_width ;
    int y_ratio = ((height) << FP_SHIFT) / new_height ;

    // LUT X per evitare ricalcoli inutili nel loop interno
    int* lut = (int*)malloc(new_width * 3 * sizeof(int));
    int* x0 = lut; //std::vector<int> x0(new_width);
    int* x1 = lut + new_width; //std::vector<int> x1(new_width);
    int* dx = lut + (new_width * 2); //std::vector<int> dx(new_width); 

    for (int x = 0; x < new_width; x++) {
        int gx = x * x_ratio;
        x0[x] = gx >> FP_SHIFT;
        dx[x] = gx & (FP_ONE - 1);                          //Q16.16
        x1[x] = (x0[x] + 1 < width - 1) ? x0[x] + 1 : width - 1;
        //std::min(x0[x] + 1, width - 1);
    }

    int64_t acc;
    
    for (int y = 0; y < new_height; y++) {
        int gy = y * y_ratio;
        int y0 = gy >> FP_SHIFT;
        int dy = gy & (FP_ONE - 1);                         //Q16.16    
        int y2 = (y0 + 1 < height - 1) ? y0 + 1 : height - 1;
        //std::min(y0 + 1, height - 1);

        const unsigned char* row0 = input + y0 * width * channels;
        const unsigned char* row1 = input + y2 * width * channels;
        unsigned char* dst = output + y * new_width * channels;

        int wy0 = FP_ONE - dy;
        int wy2 = dy;

        for (int x = 0; x < new_width; x++) {
            int wx0 = FP_ONE - dx[x];
            int wx1 = dx[x];

            // Pesi risultanti sono in Q32 (16+16)
            // Usiamo int64_t per evitare overflow durante la somma
            int64_t w00 = (int64_t)wx0 * wy0;
            int64_t w10 = (int64_t)wx1 * wy0;
            int64_t w01 = (int64_t)wx0 * wy2;
            int64_t w11 = (int64_t)wx1 * wy2;

            int i00 = x0[x] * channels;
            int i10 = x1[x] * channels;
            int o   = x * channels;

                // Calcolo pesato: (Valore * Peso)          //pixel(8 bit) * peso(Q32) = Q40
                int64_t acc = (int64_t)row0[i00 + 0] * w00 +
                              (int64_t)row0[i10 + 0] * w10 +
                              (int64_t)row1[i00 + 0] * w01 +
                              (int64_t)row1[i10 + 0] * w11;
                // Shift di 32 per tornare da Q32.32 a intero
                dst[o + 0] = (unsigned char)(acc >> 32); //troncamento
                        acc = (int64_t)row0[i00 + 1] * w00 +
                              (int64_t)row0[i10 + 1] * w10 +
                              (int64_t)row1[i00 + 1] * w01 +
                              (int64_t)row1[i10 + 1] * w11;
                // Shift di 32 per tornare da Q32.32 a intero
                dst[o + 1] = (unsigned char)(acc >> 32); //troncamento
                        acc = (int64_t)row0[i00 + 2] * w00 +
                              (int64_t)row0[i10 + 2] * w10 +
                              (int64_t)row1[i00 + 2] * w01 +
                              (int64_t)row1[i10 + 2] * w11;
                // Shift di 32 per tornare da Q32.32 a intero
                dst[o + 2] = (unsigned char)(acc >> 32); //troncamento
        }
    }
    free(lut);
}

void cpu_bil_omp(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    const int FP_SHIFT = 16;
    const int FP_ONE   = 1 << FP_SHIFT;

    // Rapporti di scala
    int x_ratio = ((width ) << FP_SHIFT) / new_width ;
    int y_ratio = ((height) << FP_SHIFT) / new_height ;

    // LUT X per evitare ricalcoli inutili nel loop interno
    int* lut = (int*)malloc(new_width * 3 * sizeof(int));
    int* x0 = lut; //std::vector<int> x0(new_width);
    int* x1 = lut + new_width; //std::vector<int> x1(new_width);
    int* dx = lut + (new_width * 2); //std::vector<int> dx(new_width); 

    for (int x = 0; x < new_width; x++) {
        int gx = x * x_ratio;
        x0[x] = gx >> FP_SHIFT;
        dx[x] = gx & (FP_ONE - 1);                          //Q16.16
        x1[x] = (x0[x] + 1 < width - 1) ? x0[x] + 1 : width - 1;
        //std::min(x0[x] + 1, width - 1);
    }
    #pragma omp parallel for
    for (int y = 0; y < new_height; y++) {
        int gy = y * y_ratio;
        int y0 = gy >> FP_SHIFT;
        int dy = gy & (FP_ONE - 1);                         //Q16.16    
        int y2 = (y0 + 1 < height - 1) ? y0 + 1 : height - 1;
        //std::min(y0 + 1, height - 1);

        const unsigned char* row0 = input + y0 * width * channels;
        const unsigned char* row1 = input + y2 * width * channels;
        unsigned char* dst = output + y * new_width * channels;

        int wy0 = FP_ONE - dy;
        int wy2 = dy;

        for (int x = 0; x < new_width; x++) {
            int wx0 = FP_ONE - dx[x];
            int wx1 = dx[x];

            // Pesi risultanti sono in Q32 (16+16)
            // Usiamo int64_t per evitare overflow durante la somma
            int64_t w00 = (int64_t)wx0 * wy0;
            int64_t w10 = (int64_t)wx1 * wy0;
            int64_t w01 = (int64_t)wx0 * wy2;
            int64_t w11 = (int64_t)wx1 * wy2;

            int i00 = x0[x] * channels;
            int i10 = x1[x] * channels;
            int o   = x * channels;

           int64_t acc = (int64_t)row0[i00 + 0] * w00 +
                              (int64_t)row0[i10 + 0] * w10 +
                              (int64_t)row1[i00 + 0] * w01 +
                              (int64_t)row1[i10 + 0] * w11;
                // Shift di 32 per tornare da Q32.32 a intero
            dst[o + 0] = (unsigned char)(acc >> 32); //troncamento
                    acc = (int64_t)row0[i00 + 1] * w00 +
                            (int64_t)row0[i10 + 1] * w10 +
                            (int64_t)row1[i00 + 1] * w01 +
                            (int64_t)row1[i10 + 1] * w11;
            // Shift di 32 per tornare da Q32.32 a intero
            dst[o + 1] = (unsigned char)(acc >> 32); //troncamento
                    acc = (int64_t)row0[i00 + 2] * w00 +
                            (int64_t)row0[i10 + 2] * w10 +
                            (int64_t)row1[i00 + 2] * w01 +
                            (int64_t)row1[i10 + 2] * w11;
            // Shift di 32 per tornare da Q32.32 a intero
            dst[o + 2] = (unsigned char)(acc >> 32); //troncamento
        }
    }
    free(lut);
}

// Bilinear RGB con OpenMP e ottimizzazioni fixed-point e LUT  Q8.8 (SIMD auto)
void cpu_bil_omp_v2(
    unsigned char* input, unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    if (channels != 3) return; // Ottimizziamo specificamente per RGB

    const int FP_SHIFT = 8; // Passiamo a 8 bit per i pesi
    const int FP_ONE = 1 << FP_SHIFT;

    int x_ratio = ((width ) << 16) / (new_width );
    int y_ratio = ((height ) << 16) / (new_height);
    
    int* lut = (int*)malloc(new_width * 3 * sizeof(int));
    int* x0 = lut; //std::vector<int> x0(new_width);
    int* x1 = lut + new_width; //std::vector<int> x1(new_width);
    int* dx = lut + (new_width * 2); //std::vector<int> dx(new_width); 

    for (int x = 0; x < new_width; x++) {

        int gx = x * x_ratio; // 16.16 
        int ix = gx >> 16; // indice pixel 0..width-1 
        int ix1 = (ix + 1 < width - 1) ? ix + 1 : width - 1; 

        x0[x] = ix * 3; // indice byte per RGB 
        x1[x] = ix1 * 3; // indice byte per RGB 
        dx[x] = (gx & 0xFFFF) >> 8; // peso orizzontale 0..255
    }

    #pragma omp parallel for 
    for (int y = 0; y < new_height; y++) {
        int gy = y * y_ratio;
        int y0 = gy >> 16;
        int dy = (gy & 0xFFFF) >> 8; // Peso verticale 0-255
        int y2 = (y0 + 1 < height - 1) ? y0 + 1 : height - 1; 

        const unsigned char* r0 = input + y0 * width * 3;
        const unsigned char* r1 = input + y2 * width * 3;
        unsigned char* dst = output + y * new_width * 3;

        int32_t wy0 = 256 - dy;
        int32_t wy2 = dy;

        for (int x = 0; x < new_width; x++) {
            int32_t wx1 = dx[x];                    //0..255
            int32_t wx0 = 256 - wx1;
        
            int i0 = x0[x];
            int i1 = x1[x];
            int o = x * channels;

            // Calcoliamo R, G, B separatamente ma senza loop.
            // Unrolling del loop sui 3 canali (R,G,B) 
            // Interpolazione orizzontale (riga 0 e riga 1)

            int32_t h0_0 = (r0[i0 + 0] * wx0 + r0[i1 + 0] * wx1) >> 8; 
            int32_t h1_0 = (r1[i0 + 0] * wx0 + r1[i1 + 0] * wx1) >> 8; 
            dst[o + 0] = (unsigned char)((h0_0 * wy0 + h1_0 * wy2) >> 8);

            int32_t h0_1 = (r0[i0 + 1] * wx0 + r0[i1 + 1] * wx1) >> 8; 
            int32_t h1_1 = (r1[i0 + 1] * wx0 + r1[i1 + 1] * wx1) >> 8; 
            dst[o + 1] = (unsigned char)((h0_1 * wy0 + h1_1 * wy2) >> 8);
            
            int32_t h0_2 = (r0[i0 + 2] * wx0 + r0[i1 + 2] * wx1) >> 8; 
            int32_t h1_2 = (r1[i0 + 2] * wx0 + r1[i1 + 2] * wx1) >> 8; 
            dst[o + 2] = (unsigned char)((h0_2 * wy0 + h1_2 * wy2) >> 8);
        }
    }
    free(lut);
}

#include <emmintrin.h>
#include <immintrin.h>


/*
Elabora 1 pixel alla volta. Usa i registri SSE2 solo per parallelizzare i canali R, G, B dello stesso pixel. 
È un miglioramento rispetto al codice scalare, ma lascia i registri SSE2 per metà inutilizzati (usa solo 3-6 byte su 16 disponibili).
*/

void cpu_bil_sse2(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    const int FP_SHIFT = 16;
    const int FP_ONE   = 1 << FP_SHIFT;
    const int WEIGHT_ONE = 1 << 8;

    int x_ratio = ((width )  << FP_SHIFT) / new_width ;
    int y_ratio = ((height) << FP_SHIFT) / new_height;

    int* lut = (int*)_mm_malloc(new_width * 3 * sizeof(int), 16);
    int* x0 = lut; //std::vector<int> x0(new_width);
    int* x1 = lut + new_width; //std::vector<int> x1
    int* dx = lut + (new_width * 2); //std::vector<int> dx(new_width);

    for (int x = 0; x < new_width; x++) {
        int gx = x * x_ratio;
        x0[x] = gx >> FP_SHIFT;
        dx[x] = (gx & (0xFFFF) >> 8);
        x1[x] = (x0[x] + 1 < width) ? x0[x] + 1 : x0[x];
    }

    const __m128i mask_ff = _mm_set1_epi32(0x000000FF); // to isolate a byte in each dword
    const __m128i zero = _mm_setzero_si128();

    for (int y = 0; y < new_height; y++) {
        int gy = y * y_ratio;
        int y0 = gy >> FP_SHIFT;
        int dy = (gy & (0xFFFF)) >> 8; // Riduciamo a 8 bit per stare in 16 bit totali (SIMD)
        int y2 = (y0 + 1 < height - 1) ? y0 + 1 : height - 1;

        unsigned char* row0 = input + y0 * width * channels;
        unsigned char* row1 = input + y2 * width * channels;
        unsigned char* dst = output + y * new_width * channels;

        // Peso verticale in 16 bit (Q8.8)
        int32_t v_wy2 = dy;
        int32_t v_wy0 = (256 - dy);
        int x=0;
        // process 4 pixels per iteration
        for (; x < new_width - 4; x+= 4) {
            // indices (pixel indices) 
            int ix0   = x0[x]; 
            int ix0_1 = x1[x]; 
            int ix1   = x0[x+1]; 
            int ix1_1 = x1[x+1]; 
            int ix2   = x0[x+2]; 
            int ix2_1 = x1[x+2]; 
            int ix3   = x0[x+3]; 
            int ix3_1 = x1[x+3];

            // dx weights for 4 pixels 
            int wx0_0 = WEIGHT_ONE - dx[x]; 
            int wx1_0 = dx[x]; 
            int wx0_1 = WEIGHT_ONE - dx[x+1]; 
            int wx1_1 = dx[x+1]; 
            int wx0_2 = WEIGHT_ONE - dx[x+2]; 
            int wx1_2 = dx[x+2]; 
            int wx0_3 = WEIGHT_ONE - dx[x+3]; 
            int wx1_3 = dx[x+3];

            
            // Load 4 source pixels for top row (p00,p10 for each target) 
            // We need top-left and top-right for each of the 4 targets. 
            // We'll build two __m128i: top_left = [p3,p2,p1,p0], top_right = [q3,q2,q1,q0] 
            /*************************************************************** */
            // load 32-bit words (RGB + spare byte) from top row
            uint32_t tl0 = *(const uint32_t*)(row0 + (size_t)ix0   * channels);
            uint32_t tr0 = *(const uint32_t*)(row0 + (size_t)ix0_1 * channels);
            uint32_t tl1 = *(const uint32_t*)(row0 + (size_t)ix1   * channels);
            uint32_t tr1 = *(const uint32_t*)(row0 + (size_t)ix1_1 * channels);
            uint32_t tl2 = *(const uint32_t*)(row0 + (size_t)ix2   * channels);
            uint32_t tr2 = *(const uint32_t*)(row0 + (size_t)ix2_1 * channels);
            uint32_t tl3 = *(const uint32_t*)(row0 + (size_t)ix3   * channels);
            uint32_t tr3 = *(const uint32_t*)(row0 + (size_t)ix3_1 * channels);

            // bottom row
            uint32_t bl0 = *(const uint32_t*)(row1 + (size_t)ix0   * channels);
            uint32_t br0 = *(const uint32_t*)(row1 + (size_t)ix0_1 * channels);
            uint32_t bl1 = *(const uint32_t*)(row1 + (size_t)ix1   * channels);
            uint32_t br1 = *(const uint32_t*)(row1 + (size_t)ix1_1 * channels);
            uint32_t bl2 = *(const uint32_t*)(row1 + (size_t)ix2   * channels);
            uint32_t br2 = *(const uint32_t*)(row1 + (size_t)ix2_1 * channels);
            uint32_t bl3 = *(const uint32_t*)(row1 + (size_t)ix3   * channels);
            uint32_t br3 = *(const uint32_t*)(row1 + (size_t)ix3_1 * channels);

            // extract bytes and horizontal interpolation scalar for 4 pixels 
            uint8_t out_r[4], out_g[4], out_b[4];
            // pixel 0 
            { 
                uint8_t tl_r = (uint8_t)(tl0 & 0xFF), tl_g = (uint8_t)((tl0 >> 8) & 0xFF), tl_b = (uint8_t)((tl0 >> 16) & 0xFF); 
                uint8_t tr_r = (uint8_t)(tr0 & 0xFF), tr_g = (uint8_t)((tr0 >> 8) & 0xFF), tr_b = (uint8_t)((tr0 >> 16) & 0xFF); 
                uint8_t bl_r = (uint8_t)(bl0 & 0xFF), bl_g = (uint8_t)((bl0 >> 8) & 0xFF), bl_b = (uint8_t)((bl0 >> 16) & 0xFF); 
                uint8_t br_r = (uint8_t)(br0 & 0xFF), br_g = (uint8_t)((br0 >> 8) & 0xFF), br_b = (uint8_t)((br0 >> 16) & 0xFF); 
                int h0r = (tl_r * wx0_0 + tr_r * wx1_0) >> 8; int h1r = (bl_r * wx0_0 + br_r * wx1_0) >> 8; 
                int vr = (h0r * v_wy0 + h1r * v_wy2) >> 8; int h0g = (tl_g * wx0_0 + tr_g * wx1_0) >> 8; 
                int h1g = (bl_g * wx0_0 + br_g * wx1_0) >> 8; int vg = (h0g * v_wy0 + h1g * v_wy2) >> 8; 
                int h0b = (tl_b * wx0_0 + tr_b * wx1_0) >> 8; int h1b = (bl_b * wx0_0 + br_b * wx1_0) >> 8; 
                int vb = (h0b * v_wy0 + h1b * v_wy2) >> 8; out_r[0] = (uint8_t)(vr < 0 ? 0 : (vr > 255 ? 255 : vr)); 
                out_g[0] = (uint8_t)(vg < 0 ? 0 : (vg > 255 ? 255 : vg)); out_b[0] = (uint8_t)(vb < 0 ? 0 : (vb > 255 ? 255 : vb)); 
            }

            // pixel 1 
            { 
                uint8_t tl_r = (uint8_t)(tl1 & 0xFF), tl_g = (uint8_t)((tl1 >> 8) & 0xFF), tl_b = (uint8_t)((tl1 >> 16) & 0xFF); 
                uint8_t tr_r = (uint8_t)(tr1 & 0xFF), tr_g = (uint8_t)((tr1 >> 8) & 0xFF), tr_b = (uint8_t)((tr1 >> 16) & 0xFF); 
                uint8_t bl_r = (uint8_t)(bl1 & 0xFF), bl_g = (uint8_t)((bl1 >> 8) & 0xFF), bl_b = (uint8_t)((bl1 >> 16) & 0xFF); 
                uint8_t br_r = (uint8_t)(br1 & 0xFF), br_g = (uint8_t)((br1 >> 8) & 0xFF), br_b = (uint8_t)((br1 >> 16) & 0xFF); 
                int h0r = (tl_r * wx0_1 + tr_r * wx1_1) >> 8; int h1r = (bl_r * wx0_1 + br_r * wx1_1) >> 8; 
                int vr = (h0r * v_wy0 + h1r * v_wy2) >> 8; int h0g = (tl_g * wx0_1 + tr_g * wx1_1) >> 8; 
                int h1g = (bl_g * wx0_1 + br_g * wx1_1) >> 8; int vg = (h0g * v_wy0 + h1g * v_wy2) >> 8; 
                int h0b = (tl_b * wx0_1 + tr_b * wx1_1) >> 8; int h1b = (bl_b * wx0_1 + br_b * wx1_1) >> 8; 
                int vb = (h0b * v_wy0 + h1b * v_wy2) >> 8; out_r[1] = (uint8_t)(vr < 0 ? 0 : (vr > 255 ? 255 : vr)); 
                out_g[1] = (uint8_t)(vg < 0 ? 0 : (vg > 255 ? 255 : vg)); out_b[1] = (uint8_t)(vb < 0 ? 0 : (vb > 255 ? 255 : vb)); 
            }
            // pixel 2 
            { 
                uint8_t tl_r = (uint8_t)(tl2 & 0xFF), tl_g = (uint8_t)((tl2 >> 8) & 0xFF), tl_b = (uint8_t)((tl2 >> 16) & 0xFF); 
                uint8_t tr_r = (uint8_t)(tr2 & 0xFF), tr_g = (uint8_t)((tr2 >> 8) & 0xFF), tr_b = (uint8_t)((tr2 >> 16) & 0xFF); 
                uint8_t bl_r = (uint8_t)(bl2 & 0xFF), bl_g = (uint8_t)((bl2 >> 8) & 0xFF), bl_b = (uint8_t)((bl2 >> 16) & 0xFF); 
                uint8_t br_r = (uint8_t)(br2 & 0xFF), br_g = (uint8_t)((br2 >> 8) & 0xFF), br_b = (uint8_t)((br2 >> 16) & 0xFF); 
                int h0r = (tl_r * wx0_2 + tr_r * wx1_2) >> 8; 
                int h1r = (bl_r * wx0_2 + br_r * wx1_2) >> 8; 
                int vr = (h0r * v_wy0 + h1r * v_wy2) >> 8; 
                int h0g = (tl_g * wx0_2 + tr_g * wx1_2) >> 8; 
                int h1g = (bl_g * wx0_2 + br_g * wx1_2) >> 8; 
                int vg = (h0g * v_wy0 + h1g * v_wy2) >> 8; 
                int h0b = (tl_b * wx0_2 + tr_b * wx1_2) >> 8; 
                int h1b = (bl_b * wx0_2 + br_b * wx1_2) >> 8; 
                int vb = (h0b * v_wy0 + h1b * v_wy2) >> 8; 
                out_r[2] = (uint8_t)(vr < 0 ? 0 : (vr > 255 ? 255 : vr)); 
                out_g[2] = (uint8_t)(vg < 0 ? 0 : (vg > 255 ? 255 : vg)); 
                out_b[2] = (uint8_t)(vb < 0 ? 0 : (vb > 255 ? 255 : vb)); 
            }
            // pixel 3 
            { 
                uint8_t tl_r = (uint8_t)(tl3 & 0xFF), tl_g = (uint8_t)((tl3 >> 8) & 0xFF), tl_b = (uint8_t)((tl3 >> 16) & 0xFF); 
                uint8_t tr_r = (uint8_t)(tr3 & 0xFF), tr_g = (uint8_t)((tr3 >> 8) & 0xFF), tr_b = (uint8_t)((tr3 >> 16) & 0xFF); 
                uint8_t bl_r = (uint8_t)(bl3 & 0xFF), bl_g = (uint8_t)((bl3 >> 8) & 0xFF), bl_b = (uint8_t)((bl3 >> 16) & 0xFF); 
                uint8_t br_r = (uint8_t)(br3 & 0xFF), br_g = (uint8_t)((br3 >> 8) & 0xFF), br_b = (uint8_t)((br3 >> 16) & 0xFF); 

                int h0r = (tl_r * wx0_3 + tr_r * wx1_3) >> 8; 
                int h1r = (bl_r * wx0_3 + br_r * wx1_3) >> 8; 
                int vr = (h0r * v_wy0 + h1r * v_wy2) >> 8; 
                int h0g = (tl_g * wx0_3 + tr_g * wx1_3) >> 8; 
                int h1g = (bl_g * wx0_3 + br_g * wx1_3) >> 8; 
                int vg = (h0g * v_wy0 + h1g * v_wy2) >> 8; 
                int h0b = (tl_b * wx0_3 + tr_b * wx1_3) >> 8; 
                int h1b = (bl_b * wx0_3 + br_b * wx1_3) >> 8; 
                int vb = (h0b * v_wy0 + h1b * v_wy2) >> 8; 
                out_r[3] = (uint8_t)(vr < 0 ? 0 : (vr > 255 ? 255 : vr)); 
                out_g[3] = (uint8_t)(vg < 0 ? 0 : (vg > 255 ? 255 : vg)); 
                out_b[3] = (uint8_t)(vb < 0 ? 0 : (vb > 255 ? 255 : vb)); 
            }
            // build 16-byte vector: bytes e0..e15 = [p0.r,p0.g,p0.b,p1.r,p1.g,p1.b,p2.r,p2.g,p2.b,p3.r,p3.g,p3.b,pad,pad,pad,pad] 
            __m128i outv = _mm_set_epi8( 0,0,0,0, // e15..e12 padding 
                (char)out_b[3], (char)out_g[3], (char)out_r[3], // e11..e9 
                (char)out_b[2], (char)out_g[2], (char)out_r[2], // e8..e6 
                (char)out_b[1], (char)out_g[1], (char)out_r[1], // e5..e3 
                (char)out_b[0], (char)out_g[0], (char)out_r[0] // e2..e0
            );
            // store 16 bytes (we only filled 12, remaining 4 bytes can be left as-is)
            // Use unaligned store to avoid alignment requirements 
            _mm_storeu_si128((__m128i*)(dst + x * 3), outv); 
        } 
        // tail 
        for (; x < new_width; ++x) { 
            int ix = x0[x]; 
            int ix1 = x1[x]; 
            int wx1 = dx[x]; 
            int wx0 = WEIGHT_ONE - wx1; 
            // top 
            const unsigned char* p00 = row0 + (size_t)ix * channels; 
            const unsigned char* p10 = row0 + (size_t)ix1 * channels; 
            const unsigned char* p01 = row1 + (size_t)ix * channels; 
            const unsigned char* p11 = row1 + (size_t)ix1 * channels; 
            for (int c = 0; c < 3; ++c) { 
                int h0 = (p00[c] * wx0 + p10[c] * wx1) >> 8; 
                int h1 = (p01[c] * wx0 + p11[c] * wx1) >> 8; 
                int val = (h0 * v_wy0 + h1 * v_wy2) >> 8; if (val < 0) val = 0; 
                else if (val > 255) val = 255; 
                dst[x*3 + c] = (unsigned char)val; 
            }
        }
    }
    _mm_free(lut);
}


#define FP_SHIFT 8
#define FP_SCALE (1 << FP_SHIFT)

void cpu_bil_sse2_v2(
    unsigned char* image,
    unsigned char* resized,
    int width, int height,
    int new_width, int new_height,
    int channels
){ 
    float scale_x = (float)(width - 1) / new_width;
    float scale_y = (float)(height - 1) / new_height;

    // 1. Pre-calcolo delle tabelle X (Orizzontali)
    // Anche questo può essere parallelizzato
    std::vector<int> x0_table(new_width);
    std::vector<int> x1_table(new_width);
    std::vector<int16_t> wx0_table(new_width);
    std::vector<int16_t> wx1_table(new_width);

    #pragma omp parallel for num_threads(16)
    for (int x = 0; x < new_width; ++x) {
        float fx = x * scale_x;
        x0_table[x] = (int)fx;
        x1_table[x] = (x0_table[x] + 1 < width) ? x0_table[x] + 1 : x0_table[x];
        float weight = fx - x0_table[x];
        wx1_table[x] = (int16_t)(weight * 256.0f);
        wx0_table[x] = 256 - wx1_table[x];
    }

    // 2. Loop principale
    for (int y = 0; y < new_height; ++y) {
        float fy = y * scale_y;
        int y0 = (int)fy;
        int y1 = (y0 + 1 < height) ? y0 + 1 : y0;
        
        int wy1_val = (int)((fy - y0) * 256.0f);
        int wy0_val = 256 - wy1_val;

        __m128i v_wy0 = _mm_set1_epi16((short)wy0_val);
        __m128i v_wy1 = _mm_set1_epi16((short)wy1_val);

        const uint8_t* row0_ptr = &image[y0 * width * channels];
        const uint8_t* row1_ptr = &image[y1 * width * channels];
        uint8_t* dst_row = &resized[y * new_width * channels];

        for (int x = 0; x < new_width; ++x) {
            int x0 = x0_table[x] * channels;
            int x1 = x1_table[x] * channels;
            
            __m128i v_wx0 = _mm_set1_epi16(wx0_table[x]);
            __m128i v_wx1 = _mm_set1_epi16(wx1_table[x]);

            // Caricamento pixel ottimizzato (RGB)
            __m128i p00 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)&row0_ptr[x0]), _mm_setzero_si128());
            __m128i p10 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)&row0_ptr[x1]), _mm_setzero_si128());
            __m128i p01 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)&row1_ptr[x0]), _mm_setzero_si128());
            __m128i p11 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)&row1_ptr[x1]), _mm_setzero_si128());

            // Interpolazione Orizzontale
            __m128i r0 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(p00, v_wx0), _mm_mullo_epi16(p10, v_wx1)), 8);
            __m128i r1 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(p01, v_wx0), _mm_mullo_epi16(p11, v_wx1)), 8);

            // Interpolazione Verticale
            __m128i res = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(r0, v_wy0), _mm_mullo_epi16(r1, v_wy1)), 8);

            // Scrittura veloce
            __m128i final = _mm_packus_epi16(res, res);
            int32_t pixel_data = _mm_cvtsi128_si32(final);
            
            // Copia dei 3 byte RGB
            dst_row[x * 3 + 0] = (uint8_t)(pixel_data & 0xFF);
            dst_row[x * 3 + 1] = (uint8_t)((pixel_data >> 8) & 0xFF);
            dst_row[x * 3 + 2] = (uint8_t)((pixel_data >> 16) & 0xFF);
        }
    }
}
/*
Introduce il concetto di elaborare 4 pixel nel loop principale, ma internamente esegue ancora 4 micro-loop SIMD separati. 
Il vantaggio qui non è nel calcolo, ma nella preparazione alla "scrittura bulk".
Q8.8 per i pesi orizzontali e verticali
*/
/*
void cpu_bil_sse2_v2(
    unsigned char* input,    // Deve essere allocato con aligned_malloc<unsigned char, 16>
    unsigned char* output,   // Deve essere allocato con aligned_malloc<unsigned char, 16>
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    if (channels != 3) return;
    const int FP_SHIFT = 16;
    const int FP_ONE   = 1 << FP_SHIFT;

    // Calcolo ratio sincronizzato con la versione naive
    int x_ratio = ((width - 1) << FP_SHIFT) / new_width ;
    int y_ratio = ((height - 1) << FP_SHIFT) / new_height ;

    // Pre-calcolo LUT
    std::vector<int> x0(new_width), x1(new_width), dx(new_width);
    for (int x = 0; x < new_width; x++) {
        int gx = x * x_ratio;
        x0[x] = gx >> FP_SHIFT;
        dx[x] = (gx & (FP_ONE - 1)) >> 8; // Scala a 0-255 per SIMD
        x1[x] = (x0[x] + 1 < width) ? x0[x] + 1 : x0[x];
    }

    const int safe_limit_4byte = width * channels - 4;
    const __m128i zero = _mm_setzero_si128();

    for (int y = 0; y < new_height; y++) {
        int gy = y * y_ratio;
        int y0 = gy >> FP_SHIFT;
        int dy = (gy & (FP_ONE - 1)) >> 8;
        int y2 = (y0 + 1 < height) ? y0 + 1 : y0;

        unsigned char* r0 = input + y0 * width * channels;
        unsigned char* r1 = input + y2 * width * channels;
        unsigned char* dst_row = output + y * new_width * channels;

        __m128i v_wy2 = _mm_set1_epi16((short)dy);
        __m128i v_wy0 = _mm_set1_epi16((short)(256 - dy));
        int x = 0;
        // 1. LOOP BATCH (4 pixel alla volta)
        for (; x < new_width-4; x+=4) {
            uint32_t pixels[4];

            // Calcoliamo 4 pixel bilineari singolarmente (usando SIMD per i canali RGB)
            for (int p = 0; p < 4; p++) {
                int curr_x = x + p;
                int i0 = x0[curr_x] * channels;
                int i1 = x1[curr_x] * channels;

                // Pesi orizzontali per questo specifico pixel
                __m128i v_wx1 = _mm_set1_epi16((short)dx[curr_x]);
                __m128i v_wx0 = _mm_set1_epi16((short)(256 - dx[curr_x]));

                // CARICAMENTO VELOCE: se non siamo vicini alla fine della riga, 
                // leggiamo 16 byte con una sola istruzione (anche se disallineati nell'offset)
                // CARICAMENTO SICURO: per gli ultimi pixel della riga
                __m128i p00 = load_pixel_rgb_sse(r0, i0, safe_limit_4byte);
                __m128i p10 = load_pixel_rgb_sse(r0, i1, safe_limit_4byte);
                __m128i p01 = load_pixel_rgb_sse(r1, i0, safe_limit_4byte);
                __m128i p11 = load_pixel_rgb_sse(r1, i1, safe_limit_4byte);

                // --- Logica SIMD Core ---
                __m128i p00_16 = _mm_unpacklo_epi8(p00, zero);
                __m128i p10_16 = _mm_unpacklo_epi8(p10, zero);
                __m128i p01_16 = _mm_unpacklo_epi8(p01, zero);
                __m128i p11_16 = _mm_unpacklo_epi8(p11, zero);

                // Interpolazione orizzontale
                __m128i h0 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(p00_16, v_wx0), _mm_mullo_epi16(p10_16, v_wx1)), 8);
                __m128i h1 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(p01_16, v_wx0), _mm_mullo_epi16(p11_16, v_wx1)), 8);

                // Interpolazione verticale
                __m128i res_16 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(h0, v_wy0), _mm_mullo_epi16(h1, v_wy2)), 8);

            // Salvataggio dei 3 byte finali
            // SALVATAGGIO FONDAMENTALE nell'array temporaneo
            pixels[p] = (uint32_t)_mm_cvtsi128_si32(_mm_packus_epi16(res_16, res_16));
            }

            // TECNICA DI SCRITTURA BULK (12 byte totali per 4 pixel)
            // Impacchettiamo i 4 pixel bilineari in 'low' e 'high'
            uint64_t low = (uint64_t)(pixels[0] & 0xFFFFFF) | 
                        ((uint64_t)(pixels[1] & 0xFFFFFF) << 24) | 
                        ((uint64_t)(pixels[2] & 0xFFFF) << 48);

            uint32_t high = (uint32_t)((pixels[2] >> 16) & 0xFF) | 
                            ((uint32_t)(pixels[3] & 0xFFFFFF) << 8);

            // Scrittura veloce in memoria
            *reinterpret_cast<uint64_t*>(&dst_row[x * 3]) = low;
            *reinterpret_cast<uint32_t*>(&dst_row[x * 3 + 8]) = high;
        }
        // TAIL LOOP: Gestione pixel rimanenti (se new_width non è multiplo di 4)
        for (; x < new_width; x++) {
            int i0 = x0[x] * channels;
            int i1 = x1[x] * channels;
            __m128i v_wx1 = _mm_set1_epi16((short)dx[x]);
            __m128i v_wx0 = _mm_set1_epi16((short)(256 - dx[x]));

            __m128i p00 = load_pixel_rgb_sse(r0, i0, safe_limit_4byte);
            __m128i p10 = load_pixel_rgb_sse(r0, i1, safe_limit_4byte);
            __m128i p01 = load_pixel_rgb_sse(r1, i0, safe_limit_4byte);
            __m128i p11 = load_pixel_rgb_sse(r1, i1, safe_limit_4byte);

            __m128i h0 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(_mm_unpacklo_epi8(p00, zero), v_wx0), _mm_mullo_epi16(_mm_unpacklo_epi8(p10, zero), v_wx1)), 8);
            __m128i h1 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(_mm_unpacklo_epi8(p01, zero), v_wx0), _mm_mullo_epi16(_mm_unpacklo_epi8(p11, zero), v_wx1)), 8);
            __m128i res_16 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(h0, v_wy0), _mm_mullo_epi16(h1, v_wy2)), 8);

            uint32_t pix = (uint32_t)_mm_cvtsi128_si32(_mm_packus_epi16(res_16, res_16));
            dst_row[x * 3]     = pix & 0xFF;
            dst_row[x * 3 + 1] = (pix >> 8) & 0xFF;
            dst_row[x * 3 + 2] = (pix >> 16) & 0xFF;
        }
    }
}

/*v3 (Dual-Pixel SIMD): Impacchetta 2 pixel completi in un unico registro (Tecnica Dual-Pixel).
Satura quasi interamente i 128 bit del registro SSE.
Dimezza il numero di istruzioni di moltiplicazione (_mm_mullo_epi16) e addizione, perché una singola istruzione opera su due pixel contemporaneamente.
Q8.8 per i pesi orizzontali e verticali
*/
/*
void cpu_bil_sse2_v3(
    unsigned char* input, 
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    if (channels != 3) return;
    const int FP_SHIFT = 16;
    const int FP_ONE = 1 << FP_SHIFT;

    int x_ratio = ((width - 1) << FP_SHIFT) / new_width ;
    int y_ratio = ((height - 1) << FP_SHIFT) / new_height;

    // LUT rimane invariata
    std::vector<int> x0(new_width), x1(new_width), dx(new_width);
    for (int x = 0; x < new_width; x++) {
        int gx = x * x_ratio;
        x0[x] = gx >> FP_SHIFT;
        dx[x] = (gx & (FP_ONE - 1)) >> 8;
        x1[x] = (x0[x] + 1 < width) ? x0[x] + 1 : x0[x];
    }

    const int safe_limit_4byte = width * channels - 4;
    const __m128i zero = _mm_setzero_si128();

    for (int y = 0; y < new_height; y++) {
        int gy = y * y_ratio;
        int y0 = gy >> FP_SHIFT;
        int dy = (gy & (FP_ONE - 1)) >> 8;
        int y2 = (y0 + 1 < height) ? y0 + 1 : y0;

        unsigned char* r0 = input + y0 * width * 3;
        unsigned char* r1 = input + y2 * width * 3;
        unsigned char* dst_row = output + y * new_width * 3;

        // Pesi verticali duplicati per 2 pixel: [y0, y0, y0, y0, y2, y2, y2, y2]
        __m128i v_wy0 = _mm_set1_epi16((short)(256 - dy));
        __m128i v_wy2 = _mm_set1_epi16((short)dy);

        int x = 0;
        // Processiamo 4 pixel alla volta, ma a coppie di 2 dentro il SIMD
        for (; x <= new_width - 4; x += 4) {
            uint32_t pixels[4];

            for (int batch = 0; batch < 2; batch++) { // Due mini-batch da 2 pixel ciascuno
                int b_x = x + batch * 2;
                
                // --- Caricamento Pixel 0 e 1 della coppia ---
                __m128i p00_a = _mm_cvtsi32_si128(*(int*)&r0[x0[b_x] * 3]);
                __m128i p00_b = _mm_cvtsi32_si128(*(int*)&r0[x0[b_x+1] * 3]);
                __m128i p10_a = _mm_cvtsi32_si128(*(int*)&r0[x1[b_x] * 3]);
                __m128i p10_b = _mm_cvtsi32_si128(*(int*)&r0[x1[b_x+1] * 3]);
                
                // ... ripetere per r1 (p01 e p11) ...
                __m128i p01_a = _mm_cvtsi32_si128(*(int*)&r1[x0[b_x] * 3]);
                __m128i p01_b = _mm_cvtsi32_si128(*(int*)&r1[x0[b_x+1] * 3]);
                __m128i p11_a = _mm_cvtsi32_si128(*(int*)&r1[x1[b_x] * 3]);
                __m128i p11_b = _mm_cvtsi32_si128(*(int*)&r1[x1[b_x+1] * 3]);

                // IMPACCHETTAMENTO: Mettiamo due pixel in un registro
                // Registro risultante: [R_a G_a B_a 0 R_b G_b B_b 0]
                __m128i row0_left  = _mm_unpacklo_epi8(_mm_unpacklo_epi32(p00_a, p00_b), zero);
                __m128i row0_right = _mm_unpacklo_epi8(_mm_unpacklo_epi32(p10_a, p10_b), zero);
                __m128i row1_left  = _mm_unpacklo_epi8(_mm_unpacklo_epi32(p01_a, p01_b), zero);
                __m128i row1_right = _mm_unpacklo_epi8(_mm_unpacklo_epi32(p11_a, p11_b), zero);

                // Carichiamo i pesi orizzontali per entrambi i pixel
                __m128i v_wx1 = _mm_set_epi16(dx[b_x+1], dx[b_x+1], dx[b_x+1], dx[b_x+1], dx[b_x], dx[b_x], dx[b_x], dx[b_x]);
                __m128i v_wx0 = _mm_set_epi16(256-dx[b_x+1], 256-dx[b_x+1], 256-dx[b_x+1], 256-dx[b_x+1], 256-dx[b_x], 256-dx[b_x], 256-dx[b_x], 256-dx[b_x]);

                // INTERPOLAZIONE (Dual Pixel)
                __m128i h0 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(row0_left, v_wx0), _mm_mullo_epi16(row0_right, v_wx1)), 8);
                __m128i h1 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(row1_left, v_wx0), _mm_mullo_epi16(row1_right, v_wx1)), 8);
                __m128i res = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(h0, v_wy0), _mm_mullo_epi16(h1, v_wy2)), 8);
                
                __m128i res_8 = _mm_packus_epi16(res, res);
                
                // Estraiamo i due pixel a 32-bit
                pixels[batch * 2]     = _mm_cvtsi128_si32(res_8);
                pixels[batch * 2 + 1] = _mm_cvtsi128_si32(_mm_srli_si128(res_8, 4));
            }

            // Scrittura Bulk (Identica a prima)
            uint64_t low = (uint64_t)(pixels[0] & 0xFFFFFF) | ((uint64_t)(pixels[1] & 0xFFFFFF) << 24) | ((uint64_t)(pixels[2] & 0xFFFF) << 48);
            uint32_t high = (uint32_t)((pixels[2] >> 16) & 0xFF) | ((uint32_t)(pixels[3] & 0xFFFFFF) << 8);
            *(uint64_t*)&dst_row[x * 3] = low;
            *(uint32_t*)&dst_row[x * 3 + 8] = high;
        }
        // TAIL LOOP: Gestione pixel rimanenti (se new_width non è multiplo di 4)
        for (; x < new_width; x++) {
            int i0 = x0[x] * channels;
            int i1 = x1[x] * channels;
            __m128i v_wx1 = _mm_set1_epi16((short)dx[x]);
            __m128i v_wx0 = _mm_set1_epi16((short)(256 - dx[x]));

            __m128i p00 = load_pixel_rgb_sse(r0, i0, safe_limit_4byte);
            __m128i p10 = load_pixel_rgb_sse(r0, i1, safe_limit_4byte);
            __m128i p01 = load_pixel_rgb_sse(r1, i0, safe_limit_4byte);
            __m128i p11 = load_pixel_rgb_sse(r1, i1, safe_limit_4byte);

            __m128i h0 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(_mm_unpacklo_epi8(p00, zero), v_wx0), _mm_mullo_epi16(_mm_unpacklo_epi8(p10, zero), v_wx1)), 8);
            __m128i h1 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(_mm_unpacklo_epi8(p01, zero), v_wx0), _mm_mullo_epi16(_mm_unpacklo_epi8(p11, zero), v_wx1)), 8);
            __m128i res_16 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(h0, v_wy0), _mm_mullo_epi16(h1, v_wy2)), 8);

            uint32_t pix = (uint32_t)_mm_cvtsi128_si32(_mm_packus_epi16(res_16, res_16));
            dst_row[x * 3]     = pix & 0xFF;
            dst_row[x * 3 + 1] = (pix >> 8) & 0xFF;
            dst_row[x * 3 + 2] = (pix >> 16) & 0xFF;
        }
    }
}

// Versione con tiling per migliorare la località di riferimento
void cpu_bil_sse2_v4(
    unsigned char* input, 
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    if (channels != 3) return;
    const int FP_SHIFT = 16;
    const int FP_ONE = 1 << FP_SHIFT;
    const int TILE_SIZE = 64; // Dimensione ottimale per Cache L1/L2

    int x_ratio = ((width - 1) << FP_SHIFT) / new_width;
    int y_ratio = ((height - 1) << FP_SHIFT) / new_height;

    // LUT pre-calcolata
    std::vector<int> x0(new_width), dx(new_width);
    for (int x = 0; x < new_width; x++) {
        int gx = x * x_ratio;
        x0[x] = (gx >> FP_SHIFT) * 3;
        dx[x] = (gx & (FP_ONE - 1)) >> 8;
    }

    const __m128i zero = _mm_setzero_si128();

    // Loop sui blocchi (Tiles)
    for (int ty = 0; ty < new_height; ty += TILE_SIZE) {

        // Calcolo limiti del tile attuale
        int y_end = std::min(ty + TILE_SIZE, new_height);

        for (int tx = 0; tx < new_width; tx += TILE_SIZE) {
            
            // Calcolo limiti del tile attuale
            int x_end = std::min(tx + TILE_SIZE, new_width);

            for (int y = ty; y < y_end; y++) {
                int gy = y * y_ratio;
                int y0_idx = gy >> FP_SHIFT;
                int dy = (gy & (FP_ONE - 1)) >> 8;
                int y2_idx = (y0_idx + 1 < height) ? y0_idx + 1 : y0_idx;

                unsigned char* r0 = input + y0_idx * width * 3;
                unsigned char* r1 = input + y2_idx * width * 3;
                unsigned char* dst_row = output + y * new_width * 3;

                __m128i v_wy0 = _mm_set1_epi16((short)(256 - dy));
                __m128i v_wy2 = _mm_set1_epi16((short)dy);

                int x = tx;
                // SIMD all'interno del tile
                int x_simd_end = tx + ((x_end - tx) / 4) * 4;

                for (; x < x_simd_end; x += 4) {
                    uint32_t pixels[4];
                    for (int batch = 0; batch < 2; batch++) {
                        int bx = x + batch * 2;
                        int i0_a = x0[bx];     int i0_b = x0[bx+1];
                        int i1_a = i0_a + 3;   int i1_b = i0_b + 3;

                        __m128i p00_a = _mm_cvtsi32_si128(*(int*)&r0[i0_a]);
                        __m128i p00_b = _mm_cvtsi32_si128(*(int*)&r0[i0_b]);
                        __m128i p10_a = _mm_cvtsi32_si128(*(int*)&r0[i1_a]);
                        __m128i p10_b = _mm_cvtsi32_si128(*(int*)&r1[i1_b]);

                        __m128i p01_a = _mm_cvtsi32_si128(*(int*)&r1[i0_a]);
                        __m128i p01_b = _mm_cvtsi32_si128(*(int*)&r1[i0_b]);
                        __m128i p11_a = _mm_cvtsi32_si128(*(int*)&r1[i1_a]);
                        __m128i p11_b = _mm_cvtsi32_si128(*(int*)&r1[i1_b]);

                        __m128i row0_L = _mm_unpacklo_epi8(_mm_unpacklo_epi32(p00_a, p00_b), zero);
                        __m128i row0_R = _mm_unpacklo_epi8(_mm_unpacklo_epi32(p10_a, p10_b), zero);
                        __m128i row1_L = _mm_unpacklo_epi8(_mm_unpacklo_epi32(p01_a, p01_b), zero);
                        __m128i row1_R = _mm_unpacklo_epi8(_mm_unpacklo_epi32(p11_a, p11_b), zero);

                        __m128i v_wx1 = _mm_set_epi16(dx[bx+1], dx[bx+1], dx[bx+1], dx[bx+1], dx[bx], dx[bx], dx[bx], dx[bx]);
                        __m128i v_wx0 = _mm_set_epi16(256-dx[bx+1], 256-dx[bx+1], 256-dx[bx+1], 256-dx[bx+1], 256-dx[bx], 256-dx[bx], 256-dx[bx], 256-dx[bx]);

                        __m128i h0 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(row0_L, v_wx0), _mm_mullo_epi16(row0_R, v_wx1)), 8);
                        __m128i h1 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(row1_L, v_wx0), _mm_mullo_epi16(row1_R, v_wx1)), 8);
                        __m128i res = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(h0, v_wy0), _mm_mullo_epi16(h1, v_wy2)), 8);
                        
                        __m128i res_8 = _mm_packus_epi16(res, res);
                        pixels[batch * 2]     = _mm_cvtsi128_si32(res_8);
                        pixels[batch * 2 + 1] = _mm_cvtsi128_si32(_mm_srli_si128(res_8, 4));
                    }

                    // Scrittura Bulk
                    uint64_t* d64 = (uint64_t*)&dst_row[x * 3];
                    d64[0] = (uint64_t)(pixels[0] & 0xFFFFFF) | ((uint64_t)(pixels[1] & 0xFFFFFF) << 24) | ((uint64_t)(pixels[2] & 0xFFFF) << 48);
                    *(uint32_t*)&dst_row[x * 3 + 8] = (uint32_t)((pixels[2] >> 16) & 0xFF) | ((uint32_t)(pixels[3] & 0xFFFFFF) << 8);
                }
                // Tail loop per il resto del tile
                for (; x < x_end; x++) { }
            }
        }
    }
}
*/

int main() {
    int width, height, channels;

    // Carichiamo l'immagine RGB
    unsigned char* image = stbi_load(
        //"./mario.png",
        "./mario.png",
        &width,
        &height,
        &channels,
        3   // forziamo RGB
    );

    if (!image) {
        printf("Errore caricamento immagine: %s\n",
            stbi_failure_reason());
        return 1;
    }

    channels = 3;

    int new_width = width * 4;
    int new_height = height * 4;

    unsigned char* resized =
        (unsigned char*)malloc(new_width * new_height * channels);

    if (!resized) {
        printf("Errore allocazione memoria\n");
        stbi_image_free(image);
        return 1;
    }

    size_t output_bytes = new_width * new_height * channels;
    size_t data_size =  sizeof(unsigned char);

    long ref_time_v1;
    long ref_time_v2;
    long ref_time_omp;
    long ref_time_omp_v2;
    long ref_time_sse2;
    long ref_time_sse2_v2;

    ref_time_v1 = time_and_print(
        "Bilinear CPU v1",
        cpu_bil_v1,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        0,   // no speedup yet
        0
    );
    stbi_write_png("resized_cpu_bil_v1.png", new_width, new_height, channels, resized, new_width * channels);

    ref_time_v2 = time_and_print(
        "Bilinear CPU v2",
        cpu_bil_v2,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        0
    );
    stbi_write_png("resized_cpu_bil_v2.png", new_width, new_height, channels, resized, new_width * channels);
    
    int threads = omp_get_max_threads();
    ref_time_omp = time_and_print(
        "Bilinear CPU omp",
        cpu_bil_omp,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,  
        threads
    );
    stbi_write_png("resized_cpu_bil_omp.png", new_width, new_height, channels, resized, new_width * channels);

    ref_time_omp_v2 = time_and_print(
        "Bilinear CPU omp_v2",
        cpu_bil_omp_v2,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        threads
    );
    stbi_write_png("resized_cpu_bil_omp_v2.png", new_width, new_height, channels, resized, new_width * channels);

    ref_time_sse2 = time_and_print(
        "Bilinear CPU sse2",
        cpu_bil_sse2,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        0
    );
    stbi_write_png("resized_cpu_bil_sse2.png", new_width, new_height, channels, resized, new_width * channels);

    ref_time_sse2_v2 = time_and_print(
        "Bilinear CPU sse2_v2",
        cpu_bil_sse2_v2,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        0
    );
    stbi_write_png("resized_cpu_bil_sse2_v2.png", new_width, new_height, channels, resized, new_width * channels);

    stbi_image_free(image);
    free(resized);

    printf("Upscaling RGB completato!\n");
    return 0;
}

