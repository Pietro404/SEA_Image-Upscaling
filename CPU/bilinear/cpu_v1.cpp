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
    float x_ratio = (float)(width - 1) / new_width;
    float y_ratio = (float)(height - 1) / new_height;

    for (int j = 0; j < new_height; j++) {
        float gy = j * y_ratio;
        int y = (int)gy;
        float dy = gy - y;
        int y1 = (y + 1 < height) ? y + 1 : y;

        for (int i = 0; i < new_width; i++) {
            float gx = i * x_ratio;
            int x = (int)gx;
            float dx = gx - x;
            int x1 = (x + 1 < width) ? x + 1 : x;

            for (int c = 0; c < channels; c++) {
                unsigned char p00 = input[(y  * width + x ) * channels + c];
                unsigned char p10 = input[(y  * width + x1) * channels + c];
                unsigned char p01 = input[(y1 * width + x ) * channels + c];
                unsigned char p11 = input[(y1 * width + x1) * channels + c];

                float value =
                    p00 * (1 - dx) * (1 - dy) +
                    p10 * dx       * (1 - dy) +
                    p01 * (1 - dx) * dy       +
                    p11 * dx       * dy;

                output[(j * new_width + i) * channels + c] =
                    (unsigned char)(value + 0.5f); // arrotondamento
            }
        }
    }
}

#include <algorithm>
#define FP_SHIFT 16
#define FP_ONE (1 << FP_SHIFT)
#define FP_HALF (1 << (FP_SHIFT - 1))
#define LUT_SIZE 1024 // Risoluzione della frazione (10 bit)
#define LUT_SHIFT (FP_SHIFT - 10) 

// Tabella pre-calcolata (da inizializzare una volta con interi)
int BIC_LUT_INT[2048]; // Copre l'intervallo [0, 2.0]

void init_bicubic_lut_int() {
    float a = -0.5f;
    for (int i = 0; i < 2048; i++) {
        float d = (float)i / 1024.0f; // d da 0 a 2
        float w;
        if (d <= 1.0f) w = (a + 2.0f)*(d*d*d) - (a + 3.0f)*(d*d) + 1.0f;
        else if (d < 2.0f) w = a*(d*d*d) - 5.0f*a*(d*d) + 8.0f*a*d - 4.0f*a;
        else w = 0.0f;
        BIC_LUT_INT[i] = (int)(w * FP_ONE);
    }
}

void cpu_bic_v2(
    unsigned char* input, unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    // Rapporti di scala in Fixed-Point Q16.16
    // Usiamo (width-1) per mappare esattamente l'ultimo pixel
    int x_ratio = ((width - 1) << FP_SHIFT) / (new_width > 1 ? new_width - 1 : 1);
    int y_ratio = ((height - 1) << FP_SHIFT) / (new_height > 1 ? new_height - 1 : 1);

    for (int y = 0; y < new_height; y++) {
        // Coordinata Y in fixed-point
        int curr_y = y * y_ratio; 
        int iy = curr_y >> FP_SHIFT;
        int fy = (curr_y & (FP_ONE - 1)); // Parte frazionaria

        // Pesi verticali (4 pixel intorno a iy)
        int wy[4];
        // fy >> LUT_SHIFT ci dà l'indice nella tabella
        wy[0] = BIC_LUT_INT[(fy + FP_ONE) >> LUT_SHIFT]; // d = 1 + fract
        wy[1] = BIC_LUT_INT[fy >> LUT_SHIFT];            // d = fract
        wy[2] = BIC_LUT_INT[(FP_ONE - fy) >> LUT_SHIFT]; // d = 1 - fract
        wy[3] = BIC_LUT_INT[(2 * FP_ONE - fy) >> LUT_SHIFT]; // d = 2 - fract

        for (int x = 0; x < new_width; x++) {
            int curr_x = x * x_ratio;
            int ix = curr_x >> FP_SHIFT;
            int fx = (curr_x & (FP_ONE - 1));

            int wx[4];
            wx[0] = BIC_LUT_INT[(fx + FP_ONE) >> LUT_SHIFT];
            wx[1] = BIC_LUT_INT[fx >> LUT_SHIFT];
            wx[2] = BIC_LUT_INT[(FP_ONE - fx) >> LUT_SHIFT];
            wx[3] = BIC_LUT_INT[(2 * FP_ONE - fx) >> LUT_SHIFT];

            for (int c = 0; c < channels; c++) {
                long long sum = 0; // Usiamo long long per evitare overflow nelle somme parziali

                for (int m = -1; m <= 2; m++) {
                    int sy = std::clamp(iy + m, 0, height - 1);
                    int weight_y = wy[m + 1];

                    for (int n = -1; n <= 2; n++) {
                        int sx = std::clamp(ix + n, 0, width - 1);
                        int weight_x = wx[n + 1];

                        int pixel = input[(sy * width + sx) * channels + c];
                        
                        // Combinazione pesi: (W1 * W2) >> FP_SHIFT
                        int combined_w = (int)(((long long)weight_y * weight_x) >> FP_SHIFT);
                        sum += (long long)pixel * combined_w;
                    }
                }

                // Risultato finale: normalizzazione e arrotondamento
                int res = (int)((sum + FP_HALF) >> FP_SHIFT);
                output[(y * new_width + x) * channels + c] = (unsigned char)std::clamp(res, 0, 255);
            }
        }
    }
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
    const int FP_HALF  = 1 << (31); // Per arrotondamento nel finale Q32

    // Rapporti di scala
    int x_ratio = ((width - 1) << FP_SHIFT) / new_width ;
    int y_ratio = ((height - 1) << FP_SHIFT) / new_height ;

    // LUT X per evitare ricalcoli inutili nel loop interno
    std::vector<int> x0(new_width), x1(new_width),dx(new_width);  
    //int* x0 = (int*)malloc(new_width * sizeof(int));
    //int* x1 = (int*)malloc(new_width * sizeof(int));
    //int* dx = (int*)malloc(new_width * sizeof(int));

    for (int x = 0; x < new_width; x++) {
        int gx = x * x_ratio;
        x0[x] = gx >> FP_SHIFT;
        dx[x] = gx & (FP_ONE - 1);
        x1[x] = (x0[x] + 1 < width) ? x0[x] + 1 : x0[x];
    }

    // Parallelizzazione sulle righe (y)
    #pragma omp parallel for
    for (int y = 0; y < new_height; y++) {
        int gy = y * y_ratio;
        int y0 = gy >> FP_SHIFT;
        int dy = gy & (FP_ONE - 1);
        int y1 = (y0 + 1 < height) ? y0 + 1 : y0;

        const unsigned char* row0 = input + y0 * width * channels;
        const unsigned char* row1 = input + y1 * width * channels;
        unsigned char* dst = output + y * new_width * channels;

        int wy0 = FP_ONE - dy;
        int wy1 = dy;

        for (int x = 0; x < new_width; x++) {
            int wx0 = FP_ONE - dx[x];
            int wx1 = dx[x];

            // Pesi risultanti sono in Q32 (16+16)
            // Usiamo int64_t per evitare overflow durante la somma
            int64_t w00 = (int64_t)wx0 * wy0;
            int64_t w10 = (int64_t)wx1 * wy0;
            int64_t w01 = (int64_t)wx0 * wy1;
            int64_t w11 = (int64_t)wx1 * wy1;

            int i00 = x0[x] * channels;
            int i10 = x1[x] * channels;
            int o   = x * channels;

            for (int c = 0; c < channels; c++) {
                // Calcolo pesato: (Valore * Peso)
                int64_t acc = (int64_t)row0[i00 + c] * w00 +
                              (int64_t)row0[i10 + c] * w10 +
                              (int64_t)row1[i00 + c] * w01 +
                              (int64_t)row1[i10 + c] * w11;

                // Shift di 32 per tornare da Q32.32 a intero
                // Aggiungiamo (1LL << 31) per un arrotondamento corretto invece del troncamento
                dst[o + c] = (unsigned char)((acc + (1LL << 31)) >> 32);
            }
        }
    }
    /* 
    free(x0);
    free(x1);
    free(dx);
    */
}

// Bilinear RGB con OpenMP e ottimizzazioni fixed-point e LUT  Q8.8 (SIMD auto)
void cpu_bil_omp_v2(
    unsigned char* input, unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    if (channels != 3) return; // Ottimizziamo specificamente per RGB

    const int FP_SHIFT = 8; // Passiamo a 8 bit per i pesi per massimizzare SIMD auto
    const int FP_ONE = 1 << FP_SHIFT;

    int x_ratio = ((width - 1) << 16) / new_width;
    int y_ratio = ((height - 1) << 16) / new_height;

    std::vector<int> x0(new_width), x1(new_width), dx(new_width);
    for (int x = 0; x < new_width; x++) {
        int gx = x * x_ratio;
        x0[x] = (gx >> 16) * 3;
        x1[x] = ((gx >> 16) + 1 < width) ? x0[x] + 3 : x0[x];
        dx[x] = (gx & 0xFFFF) >> 8; // Peso orizzontale 0-255
    }

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < new_height; y++) {
        int gy = y * y_ratio;
        int y0 = gy >> 16;
        int dy = (gy & 0xFFFF) >> 8; // Peso verticale 0-255
        int y1 = (y0 + 1 < height) ? y0 + 1 : y0;

        const unsigned char* r0 = input + y0 * width * 3;
        const unsigned char* r1 = input + y1 * width * 3;
        unsigned char* dst = output + y * new_width * 3;

        int wy1 = dy;
        int wy0 = 256 - dy;

        for (int x = 0; x < new_width; x++) {
            int wx1 = dx[x];
            int wx0 = 256 - wx1;

            int i0 = x0[x];
            int i1 = x1[x];

            // Calcoliamo R, G, B separatamente ma senza loop.
            // Usiamo int32_t: il compilatore userà istruzioni SIMD a 32-bit (4 o 8 pixel alla volta)
            for (int c = 0; c < channels; c++) {
                // Interpolazione orizzontale (riga 0 e riga 1)
                int h0 = (r0[i0 + c] * wx0 + r0[i1 + c] * wx1) >> 8;
                int h1 = (r1[i0 + c] * wx0 + r1[i1 + c] * wx1) >> 8;

                // Interpolazione verticale
                dst[x * channels + c] = (unsigned char)((h0 * wy0 + h1 * wy1) >> 8);
            }
        }
    }
}


void cpu_bil_omp_v3(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    if (channels != 3) return;

    const int FP_SHIFT = 16;
    const int FP_ONE   = 1 << FP_SHIFT;

    int x_ratio = ((width  - 1) << FP_SHIFT) / new_width;
    int y_ratio = ((height - 1) << FP_SHIFT) / new_height;

    // LUT X (Q8.8)
    std::vector<int> x0(new_width), x1(new_width), dx(new_width);
    for (int x = 0; x < new_width; x++) {
        int gx = x * x_ratio;
        x0[x] = gx >> FP_SHIFT;
        dx[x] = (gx & (FP_ONE - 1)) >> 8;   // Q8.8
        x1[x] = (x0[x] + 1 < width) ? x0[x] + 1 : x0[x];
    }

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < new_height; y++) {
        int gy = y * y_ratio;
        int y0 = gy >> FP_SHIFT;
        int dy = (gy & (FP_ONE - 1)) >> 8;  // Q8.8
        int y1 = (y0 + 1 < height) ? y0 + 1 : y0;

        unsigned char* r0 = input + y0 * width * 3;
        unsigned char* r1 = input + y1 * width * 3;
        unsigned char* dst = output + y * new_width * 3;

        int wy0 = 256 - dy;
        int wy1 = dy;

        for (int x = 0; x < new_width; x++) {
            int wx0 = 256 - dx[x];
            int wx1 = dx[x];

            int i00 = x0[x] * 3;
            int i10 = x1[x] * 3;

            // RGB
            for (int c = 0; c < 3; c++) {
                // lerp orizzontale (Q8.8 → Q16)
                int h0 =
                    r0[i00 + c] * wx0 +
                    r0[i10 + c] * wx1;

                int h1 =
                    r1[i00 + c] * wx0 +
                    r1[i10 + c] * wx1;

                h0 >>= 8;
                h1 >>= 8;

                // lerp verticale (Q8.8 → Q16)
                int v =
                    h0 * wy0 +
                    h1 * wy1;

                dst[x * 3 + c] = (unsigned char)(v >> 8);
            }
        }
    }
}



#include <emmintrin.h>
#include <immintrin.h>

template<typename T>
inline __m128i load_pixel_rgb_sse(const T* row, int idx, int safe_limit) {
    if (idx <= safe_limit) {
        // Caricamento veloce a 32-bit (4 byte)
        return _mm_cvtsi32_si128(*(const int*)&row[idx]);
    } else {
        // Caricamento sicuro byte per byte per il bordo destro
        unsigned int tmp = 0;
        tmp |= (unsigned int)row[idx];
        tmp |= ((unsigned int)row[idx + 1] << 8);
        tmp |= ((unsigned int)row[idx + 2] << 16);
        return _mm_cvtsi32_si128(tmp);
    }
}

// Utilizziamo l'allineamento 
template <typename T, size_t Alignment> 
T *aligned_malloc(const size_t size) {
    void *ptr = _mm_malloc(size * sizeof(T), Alignment);
    if (ptr == nullptr) throw std::bad_alloc();
    return static_cast<T *>(ptr);
}

/*
Elabora 1 pixel alla volta. Usa i registri SSE2 solo per parallelizzare i canali R, G, B dello stesso pixel. 
È un miglioramento rispetto al codice scalare, ma lascia i registri SSE2 per metà inutilizzati (usa solo 3-6 byte su 16 disponibili).*/

void cpu_bil_sse2(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    const int FP_SHIFT = 16;
    const int FP_ONE   = 1 << FP_SHIFT;

    int x_ratio = ((width - 1)  << FP_SHIFT) / new_width ;
    int y_ratio = ((height - 1) << FP_SHIFT) / new_height;

    std::vector<int> x0(new_width), x1(new_width), dx(new_width);

    for (int x = 0; x < new_width; x++) {
        int gx = x * x_ratio;
        x0[x] = gx >> FP_SHIFT;
        dx[x] = gx & (FP_ONE - 1);
        x1[x] = (x0[x] + 1 < width) ? x0[x] + 1 : x0[x];
    }

    const int safe_limit = width * channels - 4; // Per caricamenti sicuri
    const __m128i zero = _mm_setzero_si128();

    for (int y = 0; y < new_height; y++) {
        int gy = y * y_ratio;
        int y0 = gy >> FP_SHIFT;
        int dy = (gy & (FP_ONE - 1)) >> 8; // Riduciamo a 8 bit per stare in 16 bit totali (SIMD)
        int y1 = (y0 + 1 < height) ? y0 + 1 : y0;

        unsigned char* row0 = input + y0 * width * channels;
        unsigned char* row1 = input + y1 * width * channels;
        unsigned char* dst_row = output + y * new_width * channels;

        // Peso verticale in 16 bit (Q8.8)
        __m128i v_wy1 = _mm_set1_epi16((short)dy);
        __m128i v_wy0 = _mm_set1_epi16((short)(256 - dy));

        for (int x = 0; x < new_width; x++) {
            int cur_dx = dx[x] >> 8; // Ridotto a 8 bit
            __m128i v_wx1 = _mm_set1_epi16((short)cur_dx);
            __m128i v_wx0 = _mm_set1_epi16((short)(256 - cur_dx));

            int i0 = x0[x] * channels;
            int i1 = x1[x] * channels;

                // Carichiamo i 4 pixel sorgenti (3 byte ciascuno)
                // Usiamo una tecnica per caricare RGB in registri SIMD
                // Carichiamo 4 byte ma ne usiamo solo 3
                // usiamo il template per il caricamento sicuro
                __m128i p00 = load_pixel_rgb_sse(row0, i0 , safe_limit);
                __m128i p10 = load_pixel_rgb_sse(row0, i1 , safe_limit);
                __m128i p01 = load_pixel_rgb_sse(row1, i0 , safe_limit);
                __m128i p11 = load_pixel_rgb_sse(row1, i1, safe_limit);

                // Unpack a 16 bit
                __m128i p00_16 = _mm_unpacklo_epi8(p00, zero);
                __m128i p10_16 = _mm_unpacklo_epi8(p10, zero);
                __m128i p01_16 = _mm_unpacklo_epi8(p01, zero);
                __m128i p11_16 = _mm_unpacklo_epi8(p11, zero);

                // Interpolazione orizzontale: (p00 * wx0 + p10 * wx1) >> 8
                __m128i h0 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(p00_16, v_wx0), _mm_mullo_epi16(p10_16, v_wx1)), 8);
                __m128i h1 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(p01_16, v_wx0), _mm_mullo_epi16(p11_16, v_wx1)), 8);

                // Interpolazione verticale: (h0 * wy0 + h1 * wy1) >> 8
                __m128i res_16 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(h0, v_wy0), _mm_mullo_epi16(h1, v_wy1)), 8);

                // Riconvertiamo in 8-bit
                __m128i res_8 = _mm_packus_epi16(res_16, res_16);

                // Salvataggio dei 3 byte (RGB)
                int final_pixel = _mm_cvtsi128_si32(res_8);
                unsigned char* d = &dst_row[x * 3];
                d[0] = (unsigned char)(final_pixel & 0xFF);
                d[1] = (unsigned char)((final_pixel >> 8) & 0xFF);
                d[2] = (unsigned char)((final_pixel >> 16) & 0xFF);;
        }
    }
}

/*
Introduce il concetto di elaborare 4 pixel nel loop principale, ma internamente esegue ancora 4 micro-loop SIMD separati. 
Il vantaggio qui non è nel calcolo, ma nella preparazione alla "scrittura bulk".
Q8.8 per i pesi orizzontali e verticali
*/

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
        int y1 = (y0 + 1 < height) ? y0 + 1 : y0;

        unsigned char* r0 = input + y0 * width * channels;
        unsigned char* r1 = input + y1 * width * channels;
        unsigned char* dst_row = output + y * new_width * channels;

        __m128i v_wy1 = _mm_set1_epi16((short)dy);
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
                __m128i res_16 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(h0, v_wy0), _mm_mullo_epi16(h1, v_wy1)), 8);

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
            __m128i res_16 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(h0, v_wy0), _mm_mullo_epi16(h1, v_wy1)), 8);

            uint32_t pix = (uint32_t)_mm_cvtsi128_si32(_mm_packus_epi16(res_16, res_16));
            dst_row[x * 3]     = pix & 0xFF;
            dst_row[x * 3 + 1] = (pix >> 8) & 0xFF;
            dst_row[x * 3 + 2] = (pix >> 16) & 0xFF;
        }
    }
}

/*v3 (Dual-Pixel SIMD): È il vero salto di qualità algoritmico. Impacchetta 2 pixel completi in un unico registro (Tecnica Dual-Pixel).
Satura quasi interamente i 128 bit del registro SSE.
Dimezza il numero di istruzioni di moltiplicazione (_mm_mullo_epi16) e addizione, perché una singola istruzione opera su due pixel contemporaneamente.
Q8.8 per i pesi orizzontali e verticali
*/
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
        int y1 = (y0 + 1 < height) ? y0 + 1 : y0;

        unsigned char* r0 = input + y0 * width * 3;
        unsigned char* r1 = input + y1 * width * 3;
        unsigned char* dst_row = output + y * new_width * 3;

        // Pesi verticali duplicati per 2 pixel: [y0, y0, y0, y0, y1, y1, y1, y1]
        __m128i v_wy0 = _mm_set1_epi16((short)(256 - dy));
        __m128i v_wy1 = _mm_set1_epi16((short)dy);

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
                __m128i res = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(h0, v_wy0), _mm_mullo_epi16(h1, v_wy1)), 8);
                
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
            __m128i res_16 = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(h0, v_wy0), _mm_mullo_epi16(h1, v_wy1)), 8);

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
                int y1_idx = (y0_idx + 1 < height) ? y0_idx + 1 : y0_idx;

                unsigned char* r0 = input + y0_idx * width * 3;
                unsigned char* r1 = input + y1_idx * width * 3;
                unsigned char* dst_row = output + y * new_width * 3;

                __m128i v_wy0 = _mm_set1_epi16((short)(256 - dy));
                __m128i v_wy1 = _mm_set1_epi16((short)dy);

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
                        __m128i res = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(h0, v_wy0), _mm_mullo_epi16(h1, v_wy1)), 8);
                        
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
                for (; x < x_end; x++) { /* codice scalare... */ }
            }
        }
    }
}


int main() {
    int width, height, channels;

    // Carichiamo l'immagine RGB
    unsigned char* image = stbi_load(
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
    long ref_time_omp_v3;
    long ref_time_simd;
    long ref_time_sse2;
    long ref_time_sse2_v2;
    long ref_time_sse2_v3;
    long ref_time_sse2_v4;

    ref_time_v1 = time_and_print(
        "Bilinear CPU v1",
        cpu_bic_v1,
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

    ref_time_omp = time_and_print(
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

    ref_time_omp_v3= time_and_print(
        "Bilinear CPU omp_v3",
        cpu_bil_omp_v3,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        threads
    );
    stbi_write_png("resized_cpu_bil_omp_v3.png", new_width, new_height, channels, resized, new_width * channels);
    
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

    ref_time_sse2_v3 = time_and_print(
        "Bilinear CPU sse2_v3",
        cpu_bil_sse2_v3,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        0
    );
    stbi_write_png("resized_cpu_bil_sse2_v3.png", new_width, new_height, channels, resized, new_width * channels);

    ref_time_sse2_v4 = time_and_print(
        "Bilinear CPU sse2_v4",
        cpu_bil_sse2_v4,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        0
    );
    stbi_write_png("resized_cpu_bil_sse2_v4.png", new_width, new_height, channels, resized, new_width * channels);
/*
    ref_time_sse2 = time_and_print(
        "Nearest neighbor CPU sse2",
        cpu_nn_sse2,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        0
    );
    stbi_write_png("resized_cpu_nn_sse2.png", new_width, new_height, channels, resized, new_width * channels);
    */

    /*
    //la toglierò
    ref_time_sse2_v2 = time_and_print(
        "Nearest neighbor CPU sse2_2",
        cpu_nn_sse2_v2,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        0
    );
    stbi_write_png("resized_cpu_nn_sse2_v2.png", new_width, new_height, channels, resized, new_width * channels);
    */

    stbi_image_free(image);
    free(resized);

    printf("Upscaling RGB completato!\n");
    return 0;
}
