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

#include <cmath>
#include <algorithm>

// Funzione Kernel Bicubica (Catmull-Rom spline con a = -0.5)
float cubic_hermite(float x) {
    float a = -0.5f;
    x = std::abs(x);
    if (x <= 1.0f) {
        return (a + 2.0f) * (x * x * x) - (a + 3.0f) * (x * x) + 1.0f;
    } else if (x < 2.0f) {
        return a * (x * x * x) - 5.0f * a * (x * x) + 8.0f * a * x - 4.0f * a;
    }
    return 0.0f;
}

void cpu_bic_v1(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            // Mappa le coordinate del nuovo pixel nell'immagine originale
            float px = (x + 0.5f) * x_ratio - 0.5f;
            float py = (y + 0.5f) * y_ratio - 0.5f;

            int ix = (int)std::floor(px);
            int iy = (int)std::floor(py);

            // Per ogni canale (R, G, B, etc.)
            for (int c = 0; c < channels; c++) {
                float pixel_value = 0.0f;
                float weight_sum = 0.0f;

                // Loop sull'intorno 4x4
                for (int m = -1; m <= 2; m++) {
                    for (int n = -1; n <= 2; n++) {
                        int src_x = std::clamp(ix + n, 0, width - 1);
                        int src_y = std::clamp(iy + m, 0, height - 1);

                        float dist_x = px - (ix + n);
                        float dist_y = py - (iy + m);

                        float weight = cubic_hermite(dist_x) * cubic_hermite(dist_y);

                        int index = (src_y * width + src_x) * channels + c;
                        pixel_value += input[index] * weight;
                        weight_sum += weight;
                    }
                }

                // Normalizza e scrivi il risultato
                int out_index = (y * new_width + x) * channels + c;
                output[out_index] = (unsigned char)std::clamp(pixel_value / weight_sum, 0.0f, 255.0f);
            }
        }
    }
}

#include <cmath>
#include <algorithm>
#include <omp.h>

// Bicubica ottimizzata per OpenMP 
void cpu_bic_v2(
    unsigned char* input, unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    

    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    // Dimensione del blocco (Tile size). 16-32 è ottimo per L1/L2
    const int TILE_SIZE = 16; 

    for (int ty = 0; ty < new_height; ty += TILE_SIZE) {
        for (int tx = 0; tx < new_width; tx += TILE_SIZE) {
            
            // Processa il blocco (Tile)
            for (int y = ty; y < std::min(ty + TILE_SIZE, new_height); ++y) {
                float py = (y + 0.5f) * y_ratio - 0.5f;
                int iy = (int)std::floor(py);
                float dy = py - iy;

                // Precalcolo pesi verticali per la riga del blocco
                float wy[4];
                for (int i = 0; i < 4; i++) wy[i] = cubic_hermite(dy - (i - 1));

                for (int x = tx; x < std::min(tx + TILE_SIZE, new_width); ++x) {
                    float px = (x + 0.5f) * x_ratio - 0.5f;
                    int ix = (int)std::floor(px);
                    float dx = px - ix;

                    float wx[4];
                    for (int i = 0; i < 4; i++) wx[i] = cubic_hermite(dx - (i - 1));

                    for (int c = 0; c < channels; c++) {
                        float pixel_value = 0.0f;
                        float weight_sum = 0.0f;

                        for (int m = 0; m < 4; m++) {
                            int src_y = std::clamp(iy + m - 1, 0, height - 1);
                            float w_y = wy[m];
                            int row_offset = src_y * width * channels;

                            for (int n = 0; n < 4; n++) {
                                int src_x = std::clamp(ix + n - 1, 0, width - 1);
                                float weight = w_y * wx[n];

                                pixel_value += input[row_offset + (src_x * channels) + c] * weight;
                                weight_sum += weight;
                            }
                        }
                        
                        int out_idx = (y * new_width + x) * channels + c;
                        output[out_idx] = (unsigned char)std::clamp(pixel_value / weight_sum, 0.0f, 255.0f);
                    }
                }
            }
        }
    }
}


// Bicubica ottimizzata per OpenMP e fixed-point + tiling + RGB contiguo + padding 2x2


void cpu_bic_v3( // Bicubica con tiling e rbg contiguo
    unsigned char* input, unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels // Assumiamo sia 3 per RGB
) {

    // 1. PADDING: Creiamo un'immagine "gonfiata" di 2 pixel per lato
    int pad = 2;
    int p_width = width + 2 * pad;
    int p_height = height + 2 * pad;
    std::vector<unsigned char> padded_input(p_width * p_height * channels);

    // Copia i dati originali al centro e riempi i bordi (Edge clamping nel padding)
    for (int y = -pad; y < height + pad; ++y) {
        int src_y = ((y<0)?0:(y>height-1)?(height-1):y); //std::clamp(y, 0, height - 1); 
        for (int x = -pad; x < width + pad; ++x) {
            int src_x = ((x<0)?0:(x>width-1)?(width-1):x);//std::clamp(x, 0, width - 1); 
            for (int c = 0; c < channels; ++c) {
                padded_input[((y + pad) * p_width + (x + pad)) * channels + c] = 
                    input[(src_y * width + src_x) * channels + c];
            }
        }
    }

    const int TILE_SIZE = 16; 
    // PRE-CALCOLO COORDINATE E PESI ORIZZONTALI (Solo una volta per immagine!)
    std::vector<int> ix_arr(new_width);
    std::vector<float> wx_arr(new_width * 4);
    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    for (int x = 0; x < new_width; x++) {
        float px = (x + 0.5f) * x_ratio - 0.5f;
        ix_arr[x] = (int)std::floor(px);
        float dx = px - ix_arr[x];
        for (int i = 0; i < 4; i++) 
            wx_arr[x * 4 + i] = cubic_hermite(dx - (i - 1));
    }

    for (int y = 0; y < new_height; y++) {
        float py = (y + 0.5f) * y_ratio - 0.5f;
        int iy = (int)std::floor(py);
        float dy = py - iy;
        float wy[4];
        for (int i = 0; i < 4; i++) wy[i] = cubic_hermite(dy - (i - 1));

        // Ottimizzazione: Puntatori alle 4 righe sorgente
        const unsigned char* row_ptrs[4];
        for (int m = 0; m < 4; m++) {
            row_ptrs[m] = &padded_input[((iy + 2 + m - 1) * p_width) * channels];
        }

        for (int x = 0; x < new_width; x++) {
            int px_offset = (ix_arr[x] + 1) * channels; // ix + pad - 1
            float* current_wx = &wx_arr[x * 4];
            
            float vr = 0.5f, vg = 0.5f, vb = 0.5f;

            for (int m = 0; m < 4; m++) {
            float w_y = wy[m];
            const unsigned char* pix_ptr = row_ptrs[m] + px_offset;
            
            // Unroll manuale del loop n per massimizzare il riutilizzo dei registri
                // Evitiamo ricalcoli di indici complessi
                for (int n = 0; n < 4; n++) {
                    float w = w_y * current_wx[n];
                    vr += pix_ptr[0] * w;
                    vg += pix_ptr[1] * w;
                    vb += pix_ptr[2] * w;
                    pix_ptr += 3; // Salta al prossimo pixel RGB
                }
            }

            // Scrittura finale nel buffer di output
            // Rimosso divisione: Assumiamo weight_sum = 1.0 (proprio del kernel cubico)
            int out_idx = (y * new_width + x) * channels;

            // Branchless Clamp: Convertiamo a int e usiamo operatori ternari
            // che i compilatori moderni traducono in istruzioni CMOV o MIN/MAX
            int ir = (int)vr;
            int ig = (int)vg;
            int ib = (int)vb;

            output[out_idx]     = (unsigned char)((ir<0)?0:(ir>255)?255:ir);//std::clamp(val_r , 0.0f, 255.0f);
            output[out_idx + 1] = (unsigned char)((ig<0)?0:(ig>255)?255:ig);//std::clamp(val_g , 0.0f, 255.0f);
            output[out_idx + 2] = (unsigned char)((ib<0)?0:(ib>255)?255:ib);//std::clamp(val_b , 0.0f, 255.0f);
        }
    }
}

// LUT a 15 bit per massima precisione senza float
static int BIC_LUT_INT_V4[512 * 4];

void precompute_lut_int_v4() {
    for (int i = 0; i < 512; i++) {
        float d = i / 256.0f;
        int sum = 0;
        for (int k = -1; k <= 2; k++) {
            float x = std::abs(d - k);
            float w = 0.0f;
            if (x < 1.0f) w = 1.5f * x * x * x - 2.5f * x * x + 1.0f;
            else if (x < 2.0f) w = -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
            
            BIC_LUT_INT_V4[i * 4 + (k + 1)] = (int)(w * 32768.0f); // Q15
            sum += BIC_LUT_INT_V4[i * 4 + (k + 1)];
        }
        // Forza la somma a 32768 (1.0 in Q15)
        BIC_LUT_INT_V4[i * 4 + 1] += (32768 - sum); 
    }
}
// Fixed Point + LUT + No-Clamp
void cpu_bic_v4(unsigned char* input, unsigned char* output, 
                           int width, int height, int new_width, int new_height, int channels) {
    
    static bool init = false;
    if (!init) { precompute_lut_int_v4(); init = true; }

    // 1. PADDING (Indispensabile per eliminare i clamp interni)
    int pad = 2;
    int p_w = width + 4;
    int p_h = height + 4;
    std::vector<unsigned char> padded_input(p_w * p_h * 3);
    
    for (int y = 0; y < p_h; ++y) {
        int sy = std::max(0, std::min(height - 1, y - pad));
        for (int x = 0; x < p_w; ++x) {
            int sx = std::max(0, std::min(width - 1, x - pad));
            for(int c=0; c<3; ++c)
                padded_input[(y * p_w + x) * 3 + c] = input[(sy * width + sx) * 3 + c];
        }
    }

    const int FP_SHIFT = 16;
    int x_ratio = ((long long)width << FP_SHIFT) / new_width;
    int y_ratio = ((long long)height << FP_SHIFT) / new_height;

    for (int y = 0; y < new_height; y++) {
        int gy = y * y_ratio;
        int iy = (gy >> FP_SHIFT) + pad;
        int fy = (gy & 0xFFFF) >> 8;
        const int* wy = &BIC_LUT_INT_V4[fy * 4];

        unsigned char* d_row = output + (y * new_width * 3);

        for (int x = 0; x < new_width; x++) {
            int gx = x * x_ratio;
            int ix = (gx >> FP_SHIFT) + pad;
            int fx = (gx & 0xFFFF) >> 8;
            const int* wx = &BIC_LUT_INT_V4[fx * 4];

            int r = 0, g = 0, b = 0;

            // Unroll manuale dei 4x4 campionamenti (molto più veloce del loop m,n)
            for (int m = 0; m < 4; m++) {
                int weight_y = wy[m];
                const unsigned char* s_ptr = &padded_input[((iy + m - 1) * p_w + (ix - 1)) * 3];

                // Calcolo orizzontale pesato
                // Somma(pixel * peso_x)
                int r_sum = s_ptr[0]*wx[0] + s_ptr[3]*wx[1] + s_ptr[6]*wx[2] + s_ptr[9]*wx[3];
                int g_sum = s_ptr[1]*wx[0] + s_ptr[4]*wx[1] + s_ptr[7]*wx[2] + s_ptr[10]*wx[3];
                int b_sum = s_ptr[2]*wx[0] + s_ptr[5]*wx[1] + s_ptr[8]*wx[2] + s_ptr[11]*wx[3];

                // Accumulo verticale (Q15 * Q15 = Q30)
                r += (r_sum >> 7) * weight_y; 
                g += (g_sum >> 7) * weight_y;
                b += (b_sum >> 7) * weight_y;
            }

            // Normalizzazione finale da Q30 a 8-bit (circa >> 23 se abbiamo shiftato prima)
            // Usiamo uno shift bilanciato per restare nei 32 bit
            d_row[x * 3 + 0] = (unsigned char)std::clamp(r >> 23, 0, 255);
            d_row[x * 3 + 1] = (unsigned char)std::clamp(g >> 23, 0, 255);
            d_row[x * 3 + 2] = (unsigned char)std::clamp(b >> 23, 0, 255);
        }
    }
}


void cpu_bic_omp(unsigned char* input, unsigned char* output, 
                       int width, int height, int new_width, int new_height, int channels) {
    
    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;
    const int TILE_SIZE = 16;

    // Pre-calcolo pesi orizzontali (opzionale ma consigliato per velocità)
    std::vector<int> ix_arr(new_width);
    std::vector<float> wx_arr(new_width * 4);
    for (int x = 0; x < new_width; x++) {
        float px = (x + 0.5f) * x_ratio - 0.5f;
        ix_arr[x] = (int)std::floor(px);
        float dx = px - ix_arr[x];
        for (int i = 0; i < 4; i++) wx_arr[x * 4 + i] = cubic_hermite(dx - (i - 1));
    }

    // Parallelizziamo sui blocchi (Tiles)
    //collapse(2), OpenMP distribuisce i "quadratini" dell'immagine a tutti i core in modo bilanciato.
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int ty = 0; ty < new_height; ty += TILE_SIZE) {
        for (int tx = 0; tx < new_width; tx += TILE_SIZE) {
            
            // Buffer locale al thread per il tile (evita di bloccare la memoria globale)
            // Dimensione: un tile 16x16 ha bisogno di un intorno sorgente circa proporzionale
            // Per semplicità qui usiamo l'input diretto, ma il Tiling vero servirebbe per la cache L1.

            for (int y = ty; y < std::min(ty + TILE_SIZE, new_height); y++) {
                float py = (y + 0.5f) * y_ratio - 0.5f;
                int iy = (int)std::floor(py);
                float dy = py - iy;
                float wy[4];
                for (int i = 0; i < 4; i++) wy[i] = cubic_hermite(dy - (i - 1));

                for (int x = tx; x < std::min(tx + TILE_SIZE, new_width); x++) {
                    int ix = ix_arr[x];
                    float* c_wx = &wx_arr[x * 4];
                    float r = 0, g = 0, b = 0;

                    for (int m = 0; m < 4; m++) {
                        // Clamp verticale on-the-fly invece del padding totale
                        int sy = std::max(0, std::min(height - 1, iy + m - 1));
                        const unsigned char* row_ptr = &input[sy * width * channels];
                        
                        for (int n = 0; n < 4; n++) {
                            // Clamp orizzontale on-the-fly
                            int sx = std::max(0, std::min(width - 1, ix + n - 1));
                            float w = wy[m] * c_wx[n];
                            const unsigned char* pixel = row_ptr + (sx * channels);
                            
                            r += pixel[0] * w;
                            g += pixel[1] * w;
                            b += pixel[2] * w;
                        }
                    }

                    int out_idx = (y * new_width + x) * channels;
                    output[out_idx]   = (unsigned char)std::max(0.0f, std::min(255.0f, r));
                    output[out_idx+1] = (unsigned char)std::max(0.0f, std::min(255.0f, g));
                    output[out_idx+2] = (unsigned char)std::max(0.0f, std::min(255.0f, b));
                }
            }
        }
    }
}

#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>

// LUT a 15 bit per massima precisione senza float
static int BIC_LUT_INT[512 * 4];

void precompute_lut_int() {
    for (int i = 0; i < 512; i++) {
        float d = i / 256.0f;
        int sum = 0;
        for (int k = -1; k <= 2; k++) {
            float x = std::abs(d - k);
            float w = 0.0f;
            if (x < 1.0f) w = 1.5f * x * x * x - 2.5f * x * x + 1.0f;
            else if (x < 2.0f) w = -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
            
            BIC_LUT_INT[i * 4 + (k + 1)] = (int)(w * 32768.0f); // Q15
            sum += BIC_LUT_INT[i * 4 + (k + 1)];
        }
        // Forza la somma a 32768 (1.0 in Q15)
        BIC_LUT_INT[i * 4 + 1] += (32768 - sum); 
    }
}
// Fixed Point + LUT + No-Clamp
void cpu_bic_omp_v2(unsigned char* input, unsigned char* output, 
                           int width, int height, int new_width, int new_height, int channels) {
    
    static bool init = false;
    if (!init) { precompute_lut_int(); init = true; }

    // 1. PADDING (Indispensabile per eliminare i clamp interni)
    int pad = 2;
    int p_w = width + 4;
    int p_h = height + 4;
    std::vector<unsigned char> padded_input(p_w * p_h * 3);
    
    #pragma omp parallel for
    for (int y = 0; y < p_h; ++y) {
        int sy = std::max(0, std::min(height - 1, y - pad));
        for (int x = 0; x < p_w; ++x) {
            int sx = std::max(0, std::min(width - 1, x - pad));
            for(int c=0; c<3; ++c)
                padded_input[(y * p_w + x) * 3 + c] = input[(sy * width + sx) * 3 + c];
        }
    }

    const int FP_SHIFT = 16;
    int x_ratio = ((long long)width << FP_SHIFT) / new_width;
    int y_ratio = ((long long)height << FP_SHIFT) / new_height;

    // 2. LOOP PARALLELIZZATO
    #pragma omp parallel for schedule(guided)
    for (int y = 0; y < new_height; y++) {
        int gy = y * y_ratio;
        int iy = (gy >> FP_SHIFT) + pad;
        int fy = (gy & 0xFFFF) >> 8;
        const int* wy = &BIC_LUT_INT[fy * 4];

        unsigned char* d_row = output + (y * new_width * 3);

        for (int x = 0; x < new_width; x++) {
            int gx = x * x_ratio;
            int ix = (gx >> FP_SHIFT) + pad;
            int fx = (gx & 0xFFFF) >> 8;
            const int* wx = &BIC_LUT_INT[fx * 4];

            int r = 0, g = 0, b = 0;

            // Unroll manuale dei 4x4 campionamenti (molto più veloce del loop m,n)
            for (int m = 0; m < 4; m++) {
                int weight_y = wy[m];
                const unsigned char* s_ptr = &padded_input[((iy + m - 1) * p_w + (ix - 1)) * 3];

                // Calcolo orizzontale pesato
                // Somma(pixel * peso_x)
                int r_sum = s_ptr[0]*wx[0] + s_ptr[3]*wx[1] + s_ptr[6]*wx[2] + s_ptr[9]*wx[3];
                int g_sum = s_ptr[1]*wx[0] + s_ptr[4]*wx[1] + s_ptr[7]*wx[2] + s_ptr[10]*wx[3];
                int b_sum = s_ptr[2]*wx[0] + s_ptr[5]*wx[1] + s_ptr[8]*wx[2] + s_ptr[11]*wx[3];

                // Accumulo verticale (Q15 * Q15 = Q30)
                r += (r_sum >> 7) * weight_y; 
                g += (g_sum >> 7) * weight_y;
                b += (b_sum >> 7) * weight_y;
            }

            // Normalizzazione finale da Q30 a 8-bit (circa >> 23 se abbiamo shiftato prima)
            // Usiamo uno shift bilanciato per restare nei 32 bit
            d_row[x * 3 + 0] = (unsigned char)std::clamp(r >> 23, 0, 255);
            d_row[x * 3 + 1] = (unsigned char)std::clamp(g >> 23, 0, 255);
            d_row[x * 3 + 2] = (unsigned char)std::clamp(b >> 23, 0, 255);
        }
    }
}


#include <emmintrin.h>
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <algorithm>

// --- LUT Q7 (x128) ---
// Usiamo Q7 (128) invece di Q6 (64) per eliminare l'effetto griglia, 
// ma non Q8 (256) per evitare l'immagine rossa (overflow).
static short BIC_LUT_SSE[512 * 4]; 

void precompute_bicubic_lut_sse() {
    for (int i = 0; i < 512; i++) {
        float d = i / 256.0f;
        int sum = 0;
        int weights[4];

        // Calcolo pesi in floating point
        for (int k = -1; k <= 2; k++) {
            float x = std::abs(d - k);
            float w = 0.0f;
            if (x < 1.0f) w = 1.5f * x * x * x - 2.5f * x * x + 1.0f;
            else if (x < 2.0f) w = -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
            
            // Scala a 128 (Q7)
            weights[k + 1] = (int)(w * 128.0f);
            sum += weights[k + 1];
        }

        // CORREZIONE SOMMA: Forza la somma a 128 per evitare variazioni di luminosità
        // Distribuiamo l'errore sul peso maggiore (di solito quello centrale k=0 o k=1)
        int error = 128 - sum;
        if (error != 0) {
            // Aggiungiamo l'errore al peso centrale corrente (indice 1 o 2)
            if (d < 0.5f) weights[1] += error; 
            else weights[2] += error;
        }

        for (int k = 0; k < 4; k++) BIC_LUT_SSE[i * 4 + k] = (short)weights[k];
    }
}

// Helper (Invariato)
template<typename T>
inline __m128i load_pixel_rgb_sse(const T* row, int idx, int safe_limit) {
    if (idx <= safe_limit) return _mm_cvtsi32_si128(*(const int*)&row[idx]);
    unsigned int tmp = 0;
    // Caricamento sicuro bordi
    if (idx < safe_limit + 4) tmp |= (unsigned int)row[idx];
    if (idx + 1 < safe_limit + 4) tmp |= ((unsigned int)row[idx + 1] << 8);
    if (idx + 2 < safe_limit + 4) tmp |= ((unsigned int)row[idx + 2] << 16);
    return _mm_cvtsi32_si128(tmp);
}

void cpu_bic_sse2(unsigned char* input, unsigned char* output,
                        int width, int height, int new_width, int new_height, int channels) {
    
    // Assicurati che la LUT sia calcolata!
    precompute_bicubic_lut_sse();
    

    const int FP_SHIFT = 16;
    
    // FIX DIMENSIONE: Usiamo 'width' pieno, non 'width - 1'.
    // Questo corregge lo zoom indesiderato.
    int x_ratio = ((long long)width << FP_SHIFT) / new_width;
    int y_ratio = ((long long)height << FP_SHIFT) / new_height;

    std::vector<int> ix_arr(new_width);
    std::vector<const short*> wx_ptr(new_width);

    for (int x = 0; x < new_width; x++) {
        // Aggiungiamo mezza unità per centrare il campionamento (standard grafico)
        int gx = x * x_ratio; // Opzionale: + (x_ratio >> 1) per centraggio perfetto
        ix_arr[x] = gx >> FP_SHIFT;
        // Mappiamo 0..65535 su 0..255 per la LUT (offset 8 bit)
        wx_ptr[x] = &BIC_LUT_SSE[((gx & 0xFFFF) >> 8) * 4];
    }

    const int safe_limit = width * channels - 4;
    const __m128i zero = _mm_setzero_si128();
    
    // Arrotondamento finale: Q7 + Q7 = Q14 (16384). Metà è 8192.
    // Usiamo Q13 intermedio per sicurezza.
    const __m128i rounding = _mm_set1_epi32(1 << 13); 

    for (int y = 0; y < new_height; y++) {
        int gy = y * y_ratio;
        int iy = gy >> FP_SHIFT;
        const short* wy = &BIC_LUT_SSE[((gy & 0xFFFF) >> 8) * 4];

        __m128i v_wy01 = _mm_set1_epi32((wy[1] << 16) | (wy[0] & 0xFFFF));
        __m128i v_wy23 = _mm_set1_epi32((wy[3] << 16) | (wy[2] & 0xFFFF));

        // Clamp righe verticali
        const unsigned char* rows[4];
        for(int k=0; k<4; ++k) 
            rows[k] = input + std::max(0, std::min(height - 1, iy + k - 1)) * width * channels;
            
        unsigned char* d_row = output + y * new_width * channels;

        for (int x = 0; x < new_width; x++) {
            int ix = ix_arr[x];
            const short* wx = wx_ptr[x];
            __m128i v_wx01 = _mm_set1_epi32((wx[1] << 16) | (wx[0] & 0xFFFF));
            __m128i v_wx23 = _mm_set1_epi32((wx[3] << 16) | (wx[2] & 0xFFFF));

            __m128i h_res[4]; 

            // Clamp orizzontale base
            int base_ix = ix - 1; 

            for(int k=0; k<4; ++k) {
                // Gestione bordi più accurata per evitare segfault o letture sporche
                int idx0 = std::max(0, std::min(width-1, base_ix)) * channels;
                int idx1 = std::max(0, std::min(width-1, base_ix + 1)) * channels;
                int idx2 = std::max(0, std::min(width-1, base_ix + 2)) * channels;
                int idx3 = std::max(0, std::min(width-1, base_ix + 3)) * channels;

                __m128i p0 = load_pixel_rgb_sse(rows[k], idx0, safe_limit);
                __m128i p1 = load_pixel_rgb_sse(rows[k], idx1, safe_limit);
                __m128i p2 = load_pixel_rgb_sse(rows[k], idx2, safe_limit);
                __m128i p3 = load_pixel_rgb_sse(rows[k], idx3, safe_limit);

                __m128i sum01 = _mm_madd_epi16(_mm_unpacklo_epi16(_mm_unpacklo_epi8(p0, zero), _mm_unpacklo_epi8(p1, zero)), v_wx01);
                __m128i sum23 = _mm_madd_epi16(_mm_unpacklo_epi16(_mm_unpacklo_epi8(p2, zero), _mm_unpacklo_epi8(p3, zero)), v_wx23);
                
                // Q7 Output. Max approx 32640.
                // Pack saturebbe se superasse 32767. 
                // TRUCCO ANTI-ROSSO: Shiftiamo di 1 a destra (diviso 2) per stare sicuri nel range signed 16-bit
                // Il risultato diventa Q6 (64) effettivo.
                __m128i row_total = _mm_srai_epi32(_mm_add_epi32(sum01, sum23), 1); 
                h_res[k] = _mm_packs_epi32(row_total, zero); 
            }

            // VERTICALE
            // Input: Q6 (per via dello shift sopra). Peso: Q7.
            // Totale: Q13 (8192).
            __m128i v_final = _mm_add_epi32(
                _mm_madd_epi16(_mm_unpacklo_epi16(h_res[0], h_res[1]), v_wy01),
                _mm_madd_epi16(_mm_unpacklo_epi16(h_res[2], h_res[3]), v_wy23)
            );

            // Normalizzazione Q13 -> 8 bit
            v_final = _mm_srai_epi32(_mm_add_epi32(v_final, rounding), 13);
            
            // Pack finale con saturazione unsigned (0-255)
            int val = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi16(v_final, zero), zero));

            d_row[x * 3 + 0] = (unsigned char)(val & 0xFF);
            d_row[x * 3 + 1] = (unsigned char)((val >> 8) & 0xFF);
            d_row[x * 3 + 2] = (unsigned char)((val >> 16) & 0xFF);
        }
    }
}


void cpu_bic_sse2_v2(unsigned char* input, unsigned char* output,
                        int width, int height, int new_width, int new_height, int channels) {
  if (channels != 3) return; // Ottimizzato per RGB
  // Assicurati che la LUT sia calcolata!
    precompute_bicubic_lut_sse();

    // 1. PADDING PIÙ AMPIO PER SICUREZZA SIMD
    // Aggiungiamo 4 pixel di bordo invece di 2 per evitare che l'ultimo caricamento 
    // SIMD (che legge 4 byte alla volta) vada fuori buffer.
    int pad = 4;
    int p_w = width + pad * 2;
    int p_h = height + pad * 2;
    std::vector<unsigned char> padded_input(p_w * p_h * channels, 0);

    for (int y = 0; y < p_h; ++y) {
        int src_y = std::clamp(y - pad, 0, height - 1);
        unsigned char* dst_row = &padded_input[y * p_w * channels];
        unsigned char* src_row = &input[src_y * width * channels];
        
        for (int x = 0; x < p_w; ++x) {
            int src_x = std::clamp(x - pad, 0, width - 1);
            dst_row[x * 3 + 0] = src_row[src_x * 3 + 0];
            dst_row[x * 3 + 1] = src_row[src_x * 3 + 1];
            dst_row[x * 3 + 2] = src_row[src_x * 3 + 2];
        }
    }

    const int FP_SHIFT = 16;
    int x_ratio = ((long long)width << FP_SHIFT) / new_width;
    int y_ratio = ((long long)height << FP_SHIFT) / new_height;

    const __m128i zero = _mm_setzero_si128();
    const __m128i rounding = _mm_set1_epi32(1 << 13);

    for (int y = 0; y < new_height; y++) {
        int gy = y * y_ratio;
        int iy = (gy >> FP_SHIFT) + pad; 
        const short* wy = &BIC_LUT_SSE[((gy & 0xFFFF) >> 8) * 4];

        __m128i v_wy01 = _mm_set1_epi32((wy[1] << 16) | (wy[0] & 0xFFFF));
        __m128i v_wy23 = _mm_set1_epi32((wy[3] << 16) | (wy[2] & 0xFFFF));

        unsigned char* d_row = output + y * new_width * channels;

        for (int x = 0; x < new_width; x++) {
            int gx = x * x_ratio;
            int ix = (gx >> FP_SHIFT) + pad;
            const short* wx = &BIC_LUT_SSE[((gx & 0xFFFF) >> 8) * 4];

            __m128i v_wx01 = _mm_set1_epi32((wx[1] << 16) | (wx[0] & 0xFFFF));
            __m128i v_wx23 = _mm_set1_epi32((wx[3] << 16) | (wx[2] & 0xFFFF));

            __m128i h_res[4];
            
            for(int k = 0; k < 4; ++k) {
                // Puntatore al primo dei 4 pixel (P-1, P0, P1, P2)
                const unsigned char* p_src = &padded_input[((iy + k - 1) * p_w + (ix - 1)) * channels];

                // Caricamento RGB manuale per evitare allineamenti errati o letture sporche
                // Pixel 0
                __m128i p0 = _mm_cvtsi32_si128(*(const int*)(p_src));
                // Pixel 1 (offset 3 byte)
                __m128i p1 = _mm_cvtsi32_si128(*(const int*)(p_src + 3));
                // Pixel 2 (offset 6 byte)
                __m128i p2 = _mm_cvtsi32_si128(*(const int*)(p_src + 6));
                // Pixel 3 (offset 9 byte)
                __m128i p3 = _mm_cvtsi32_si128(*(const int*)(p_src + 9));

                __m128i p0_16 = _mm_unpacklo_epi8(p0, zero);
                __m128i p1_16 = _mm_unpacklo_epi8(p1, zero);
                __m128i p2_16 = _mm_unpacklo_epi8(p2, zero);
                __m128i p3_16 = _mm_unpacklo_epi8(p3, zero);

                __m128i sum01 = _mm_madd_epi16(_mm_unpacklo_epi16(p0_16, p1_16), v_wx01);
                __m128i sum23 = _mm_madd_epi16(_mm_unpacklo_epi16(p2_16, p3_16), v_wx23);
                
                __m128i row_total = _mm_srai_epi32(_mm_add_epi32(sum01, sum23), 1); 
                h_res[k] = _mm_packs_epi32(row_total, zero); 
            }

            __m128i v_final = _mm_add_epi32(
                _mm_madd_epi16(_mm_unpacklo_epi16(h_res[0], h_res[1]), v_wy01),
                _mm_madd_epi16(_mm_unpacklo_epi16(h_res[2], h_res[3]), v_wy23)
            );

            v_final = _mm_srai_epi32(_mm_add_epi32(v_final, rounding), 13);
            __m128i v_final_8 = _mm_packus_epi16(_mm_packus_epi16(v_final, zero), zero);
            int val = _mm_cvtsi128_si32(v_final_8);

            d_row[x * 3 + 0] = (unsigned char)(val & 0xFF);
            d_row[x * 3 + 1] = (unsigned char)((val >> 8) & 0xFF);
            d_row[x * 3 + 2] = (unsigned char)((val >> 16) & 0xFF);
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
    long ref_time_v3;
    long ref_time_v4;
    long ref_time_omp;
    long ref_time_omp_v2;
    long ref_time_omp_v3;
    long ref_time_simd;
    long ref_time_sse2;
    long ref_time_sse2_v2;
    long ref_time_sse2_v3;


    ref_time_v1 = time_and_print(
        "Bicubic CPU v1",
        cpu_bic_v1,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        0,   // no speedup yet
        0
    );
    stbi_write_png("resized_cpu_bic_v1.png", new_width, new_height, channels, resized, new_width * channels);


    ref_time_v2 = time_and_print(
        "Bicubic CPU v2",
        cpu_bic_v2,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        0
    );
    stbi_write_png("resized_cpu_bic_v2.png", new_width, new_height, channels, resized, new_width * channels);

    ref_time_v3= time_and_print(
        "Bicubic CPU v3",
        cpu_bic_v3,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        0
    );
    stbi_write_png("resized_cpu_bic_v3.png", new_width, new_height, channels, resized, new_width * channels);

    ref_time_v4= time_and_print(
        "Bicubic CPU v4",
        cpu_bic_v4,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        0
    );
    stbi_write_png("resized_cpu_bic_v4.png", new_width, new_height, channels, resized, new_width * channels);

    int threads = omp_get_max_threads();
    ref_time_omp = time_and_print(
        "Bicubic CPU omp",
        cpu_bic_omp,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,  
        threads
    );
    stbi_write_png("resized_cpu_bic_omp.png", new_width, new_height, channels, resized, new_width * channels);

    ref_time_omp_v2 = time_and_print(
        "Bicubic CPU omp v2",
        cpu_bic_omp_v2,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,  
        threads
    );
    stbi_write_png("resized_cpu_bic_omp_v2.png", new_width, new_height, channels, resized, new_width * channels);

    ref_time_sse2 = time_and_print(
        "Bicubic CPU sse2",
        cpu_bic_sse2,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        0
    );
    stbi_write_png("resized_cpu_bic_sse2.png", new_width, new_height, channels, resized, new_width * channels);

    ref_time_sse2_v2 = time_and_print(
        "Bicubic CPU sse2 v2",
        cpu_bic_sse2_v2,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        0
    );
    stbi_write_png("resized_cpu_bic_sse2_v2.png", new_width, new_height, channels, resized, new_width * channels);



    stbi_image_free(image);
    free(resized);

    printf("Upscaling RGB completato!\n");
    return 0;
}
