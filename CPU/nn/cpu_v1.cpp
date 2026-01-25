#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "test.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <vector>



// Nearest Neighbor RGB
void cpu_nn_v1(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    for (int y = 0; y < new_height; y++) {
        int src_y = (int)(y * y_ratio);
        if (src_y >= height) src_y = height - 1;

        for (int x = 0; x < new_width; x++) {
            int src_x = (int)(x * x_ratio);
            if (src_x >= width) src_x = width - 1;

            int in_idx  = (src_y * width + src_x) * channels;
            int out_idx = (y * new_width + x) * channels;

            for (int c = 0; c < channels; c++)
                output[out_idx + c] = input[in_idx + c];
        }
    }
}

void cpu_nn_v2(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int new_width,
    int new_height,
    int channels
) {
    // Usiamo il fixed-point a 16 bit (Shift di 16) per non convertire in float
    // Questo permette di calcolare gli indici con semplici operazioni intere
    const int FP_SHIFT = 16;
    const int FP_ONE = 1 << FP_SHIFT;

    int x_ratio_fp = (int)((width << FP_SHIFT) / new_width);
    int y_ratio_fp = (int)((height << FP_SHIFT) / new_height);

    // 1. Pre-calcolo degli indici X (Look-up Table locale)
    // Questo elimina la moltiplicazione float e il cast dal loop interno
    std::vector<int> x_indices(new_width); //int* x_indices = (int*)malloc(new_width * sizeof(int)); 
    for (int x = 0; x < new_width; x++) {
        int s_x = (int)(x * x_ratio_fp) >> FP_SHIFT;
        if (s_x >= width) s_x = width - 1;
        x_indices[x] = s_x * channels; 
    }
    
// 2. Loop principale ottimizzato
    for (int y = 0; y < new_height; y++) {
        int src_y = (int)(y * y_ratio_fp) >> FP_SHIFT;
        if (src_y >= height) src_y = height - 1;

        // putnatori all'inizio della riga concorrente per input e output
        //CPU usa la cache L1 invece di ram
        unsigned char* src_row = &input[src_y * width * channels];
        unsigned char* dst_row = &output[y * new_width * channels];

        for (int x = 0; x < new_width; x++) {
            int src_pix_offset = x_indices[x];
            int dst_pix_offset = x * channels;

            // Ottimizzazione per canali (RGB)
            if (channels == 3) {
                dst_row[dst_pix_offset+0] = src_row[src_pix_offset+ 0];
                dst_row[dst_pix_offset+1] = src_row[src_pix_offset+ 1];
                dst_row[dst_pix_offset+2] = src_row[src_pix_offset+ 2];
            } else {
                for (int c = 0; c < channels; c++) {
                    dst_row[dst_pix_offset + c] = src_row[src_pix_offset + c];
                }
            }
        }
    }
    //free(x_indices);
}


/*
void cpu_nn_block(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;
    int block_size = 64; // blocco di righe

    for (int by = 0; by < new_height; by += block_size) {
        int y_max = (by + block_size < new_height) ? (by + block_size) : new_height;
        for (int y = by; y < y_max; y++) {
            int src_y = (int)(y * y_ratio);
            if (src_y >= height) src_y = height - 1;

            for (int x = 0; x < new_width; x++) {
                int src_x = (int)(x * x_ratio);
                if (src_x >= width) src_x = width - 1;

                int in_idx = (src_y * width + src_x) * channels;
                int out_idx = (y * new_width + x) * channels;

                for (int c = 0; c < channels; c++)
                    output[out_idx + c] = input[in_idx + c];
            }
        }
    }
}
    */

void cpu_nn_omp(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int new_width,
    int new_height,
    int channels
) {
    // Usiamo il fixed-point a 16 bit (Shift di 16)
    // Questo permette di calcolare gli indici con semplici operazioni intere
    const int FP_SHIFT = 16;
    const int FP_ONE = 1 << FP_SHIFT;

    int x_ratio_fp = (int)((width << FP_SHIFT) / new_width);
    int y_ratio_fp = (int)((height << FP_SHIFT) / new_height);

    // 1. Pre-calcolo degli indici X (Look-up Table locale)
    // Questo elimina la moltiplicazione float e il cast dal loop interno
    std::vector<int> x_indices(new_width); //int* x_indices = (int*)malloc(new_width * sizeof(int));
    
    for (int x = 0; x < new_width; x++) {
        int s_x = (int)(x * x_ratio_fp) >> FP_SHIFT;
        if (s_x >= width) s_x = width - 1;
        x_indices[x] = s_x * channels; 
    }
    
// 2. Loop principale ottimizzato
#pragma omp parallel for
    for (int y = 0; y < new_height; y++) {
        int src_y = (int)(y * y_ratio_fp) >> FP_SHIFT;
        if (src_y >= height) src_y = height - 1;

        // putnatori all'inizio della riga concorrente per input e output
        //CPU usa la cache L1 invece di ram
        unsigned char* src_row = &input[src_y * width * channels];
        unsigned char* dst_row = &output[y * new_width * channels];

        for (int x = 0; x < new_width; x++) {
            int src_pix_offset = x_indices[x];
            int dst_pix_offset = x * channels;

            // Ottimizzazione per canali (RGB)
            if (channels == 3) {
                dst_row[dst_pix_offset+0] = src_row[src_pix_offset+ 0];
                dst_row[dst_pix_offset+1] = src_row[src_pix_offset+ 1];
                dst_row[dst_pix_offset+2] = src_row[src_pix_offset+ 2];
            } else {
                for (int c = 0; c < channels; c++) {
                    dst_row[dst_pix_offset + c] = src_row[src_pix_offset + c];
                }
            }
        }
    }
    //free(x_indices);
}
/*
void cpu_nn_omp(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

#pragma omp parallel for
    for (int y = 0; y < new_height; y++) {
        int src_y = (int)(y * y_ratio);
        if (src_y >= height) src_y = height - 1;

        for (int x = 0; x < new_width; x++) {
            int src_x = (int)(x * x_ratio);
            if (src_x >= width) src_x = width - 1;

            int in_idx  = (src_y * width + src_x) * channels;
            int out_idx = (y * new_width + x) * channels;

            for (int c = 0; c < channels; c++)
                output[out_idx + c] = input[in_idx + c];
        }
    }
}
*/



#include <emmintrin.h>

/*void cpu_nn_simd(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    for (int y = 0; y < new_height; y++) {
        int src_y = (int)(y * y_ratio);
        if (src_y >= height) src_y = height - 1;

        for (int x = 0; x < new_width; x++) {
            int src_x = (int)(x * x_ratio);
            if (src_x >= width) src_x = width - 1;

            int in_idx  = (src_y * width + src_x) * 3;
            int out_idx = (y * new_width + x) * 3;

            // carico 16 byte non allineati
            __m128i pix = _mm_loadu_si128(
                (__m128i const*)(input + in_idx)
            );

            // store temporaneo
            unsigned char tmp[16];
            _mm_storeu_si128((__m128i*)tmp, pix);

            // copio solo RGB
            output[out_idx + 0] = tmp[0];
            output[out_idx + 1] = tmp[1];
            output[out_idx + 2] = tmp[2];
        }
    }
}
*/

// Utilizziamo l'allineamento 
template <typename T, size_t Alignment> 
T *aligned_malloc(const size_t size) {
    void *ptr = _mm_malloc(size * sizeof(T), Alignment);
    if (ptr == nullptr) throw std::bad_alloc();
    return static_cast<T *>(ptr);
}
// Nearest Neighbor presenta accessi regolari e indipendenti, risultando particolarmente adatto alla vettorizzazione SIMD.
//caso rgba sarebbe molto meglio parallelizzare, noi ci concentriamo su rgb
void cpu_nn_sse2(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int new_width,
    int new_height,
    int channels
) {
    // Implementazione SOLO RGB
    if (channels != 3) return ;

    const int FP_SHIFT = 16;
    const int alignment = 16;

    int x_ratio_fp = (width  << FP_SHIFT) / new_width;
    int y_ratio_fp = (height << FP_SHIFT) / new_height;

    // 1. Pre-calcolo indici allineati (LUT)
    int x = 0;
    int* x_indices = aligned_malloc<int, alignment>(new_width);
    for (; x < new_width; x++) {
        int s_x = (int)((long long)x * x_ratio_fp) >> FP_SHIFT;
        if (s_x >= width) s_x = width - 1;
        x_indices[x] = s_x * channels;
    }

    for (int y = 0; y < new_height; y++) {
        int src_y = (y * y_ratio_fp) >> FP_SHIFT;
        if (src_y >= height) src_y = height - 1;

        unsigned char* src_row = &input[src_y * width * channels];
        unsigned char* dst_row = &output[y * new_width * channels];

        x = 0;

        // SIMD: 4 pixel RGB per iterazione
        for (; x <= new_width - 4; x += 4) {

            // Carichiamo 4 indici contemporaneamente dalla LUT allineata
            __m128i v_offsets = _mm_load_si128(reinterpret_cast<const __m128i*>(&x_indices[x]));

            // Estraiamo gli indici (SSE2 non ha gather, quindi l'accesso alla memoria sorgente resta scalare)
            uint32_t off0 = _mm_cvtsi128_si32(v_offsets);
            uint32_t off1 = _mm_cvtsi128_si32(_mm_srli_si128(v_offsets, 4));
            uint32_t off2 = _mm_cvtsi128_si32(_mm_srli_si128(v_offsets, 8));
            uint32_t off3 = _mm_cvtsi128_si32(_mm_srli_si128(v_offsets, 12));

            // Carichiamo i pixel sorgente (leggiamo 4 byte ma ne useremo 3)
            uint32_t p0 = *reinterpret_cast<uint32_t*>(&src_row[off0]); // R0 G0 B0 XX
            uint32_t p1 = *reinterpret_cast<uint32_t*>(&src_row[off1]); // R1 G1 B1 XX
            uint32_t p2 = *reinterpret_cast<uint32_t*>(&src_row[off2]); // R2 G2 B2 XX    
            uint32_t p3 = *reinterpret_cast<uint32_t*>(&src_row[off3]); // R3 G3 B3 XX

            // Impacchettamento RGB: creiamo una sequenza contigua di 12 byte
            // Registro low: [P0_R, P0_G, P0_B, P1_R, P1_G, P1_B, P2_R, P2_G] (8 byte)
            uint64_t low = (uint64_t)(p0 & 0xFFFFFF) | 
                           ((uint64_t)(p1 & 0xFFFFFF) << 24) | 
                           ((uint64_t)(p2 & 0xFFFF) << 48);

            // Registro high: [P2_B, P3_R, P3_G, P3_B] (4 byte)
            // Prendiamo il terzo byte di p2 (B2) e i 3 byte di p3
            uint32_t high = (uint32_t)((p2 >> 16) & 0xFF) | 
                            ((uint32_t)(p3 & 0xFFFFFF) << 8);

            // Scrittura bulk: una da 64-bit e una da 32-bit (molto più veloce di 12 scritture da 8-bit)
            *reinterpret_cast<uint64_t*>(&dst_row[x * 3]) = low;
            *reinterpret_cast<uint32_t*>(&dst_row[x * 3 + 8]) = high;
        }

        // Tail loop per rimasugli
        for (; x < new_width; x++) {
            int src_off = x_indices[x];
            int dst_off = x * 3;
            dst_row[dst_off]   = src_row[src_off];
            dst_row[dst_off+1] = src_row[src_off+1];
            dst_row[dst_off+2] = src_row[src_off+2];
        }
    }
    _mm_free(x_indices);
}


void cpu_nn_sse2_v2(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int new_width,
    int new_height,
    int channels
) {
    if(channels !=3) {
        //print errore
        return;
    }
    const int FP_SHIFT = 16;
    const int alignment = 16;

    int x_ratio_fp = (int)((width << FP_SHIFT) / new_width);
    int y_ratio_fp = (int)((height << FP_SHIFT) / new_height);

    // Registri costanti per il calcolo SIMD
    __m128i v_x_ratio = _mm_set1_epi32(x_ratio_fp);
    __m128i v_width_m1 = _mm_set1_epi32(width - 1);
    __m128i v_base_x = _mm_setr_epi32(0, 1, 2, 3); // [0, 1, 2, 3]
    
// 2. Loop principale ottimizzato
    for (int y = 0; y < new_height; y++) {
        int src_y = (int)(y * y_ratio_fp) >> FP_SHIFT;
        if (src_y >= height) src_y = height - 1;

        // putnatori all'inizio della riga concorrente per input e output
        //CPU usa la cache L1 invece di ram
        unsigned char* src_row = &input[src_y * width * channels];
        unsigned char* dst_row = &output[y * new_width * channels];

        int x = 0;
        // Processiamo 4 pixel RGB alla volta (12 byte)
        for (; x <= new_width-4; x+=4) {
            // 1. Generiamo i valori correnti di x: [x, x+1, x+2, x+3]
            __m128i v_curr_x = _mm_add_epi32(_mm_set1_epi32(x), v_base_x);

            // 2. Calcolo s_x = (x * x_ratio_fp) >> 16
            // SSE2 mul_epi32 moltiplica solo i 2 elementi bassi a 64bit, 
            // Moltiplichiamo separatamente gli elementi pari e dispari.
            __m128i tmp1 = _mm_mul_epu32(v_curr_x, v_x_ratio); // Moltiplica elem 0 e 2
            __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(v_curr_x, 4), _mm_srli_si128(v_x_ratio, 4)); // Elem 1 e 3
           
            // Ricomponiamo il risultato spostando i bit per lo shift a 16
            tmp1 = _mm_srli_epi64(tmp1, FP_SHIFT);
            tmp2 = _mm_srli_epi64(tmp2, FP_SHIFT);

           // Shuffle per rimettere i 4 risultati in un unico registro __m128i
            __m128i v_sx = _mm_unpacklo_epi32(
                _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)),
                _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0))
            );
            v_sx = _mm_shuffle_epi32(v_sx, _MM_SHUFFLE(3, 1, 2, 0));

            // 3. Clamp (v_sx = min(v_sx, width-1))
            __m128i mask = _mm_cmpgt_epi32(v_sx, v_width_m1);
            v_sx = _mm_or_si128(_mm_andnot_si128(mask, v_sx), _mm_and_si128(mask, v_width_m1));

            // 4. Offset = s_x * 3 (Facciamo s_x * 2 + s_x )
            __m128i v_offsets = _mm_add_epi32(_mm_slli_epi32(v_sx, 1), v_sx);

           // 5. Estrazione 
            uint32_t off0 = _mm_cvtsi128_si32(v_offsets);
            uint32_t off1 = _mm_cvtsi128_si32(_mm_srli_si128(v_offsets, 4));
            uint32_t off2 = _mm_cvtsi128_si32(_mm_srli_si128(v_offsets, 8));
            uint32_t off3 = _mm_cvtsi128_si32(_mm_srli_si128(v_offsets, 12));

            // 6. Scrittura 12 byte (4 pixel RGB)
            unsigned char* d = dst_row + x * 3;

            d[0]  = src_row[off0];
            d[1]  = src_row[off0 + 1];
            d[2]  = src_row[off0 + 2];

            d[3]  = src_row[off1];
            d[4]  = src_row[off1 + 1];
            d[5]  = src_row[off1 + 2];

            d[6]  = src_row[off2];
            d[7]  = src_row[off2 + 1];
            d[8]  = src_row[off2 + 2];

            d[9]  = src_row[off3];
            d[10] = src_row[off3 + 1];
            d[11] = src_row[off3 + 2];
        }

            // tail
        for (; x < new_width; x++) {
            int sx = (x * x_ratio_fp) >> FP_SHIFT;
            if (sx >= width) sx = width - 1;

            unsigned char* s = src_row + sx * 3;
            unsigned char* d = dst_row + x * 3;

            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
        }
    }
    //_mm_free(x_indices);
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
    long ref_time_simd;
    long ref_time_sse2;
    long ref_time_sse2_v2;

    ref_time_v1 = time_and_print(
        "Nearest neighbor CPU v1",
        cpu_nn_v1,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        0,   // no speedup yet
        0
    );
    stbi_write_png("resized_cpu_nn_v1.png", new_width, new_height, channels, resized, new_width * channels);


    ref_time_v2 = time_and_print(
        "Nearest neighbor CPU v2",
        cpu_nn_v2,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        0
    );
    stbi_write_png("resized_cpu_nn_v2.png", new_width, new_height, channels, resized, new_width * channels);

    int threads = omp_get_max_threads();
    ref_time_omp = time_and_print(
        "Nearest neighbor CPU omp",
        cpu_nn_omp,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,  
        threads
    );
    stbi_write_png("resized_cpu_nn_omp.png", new_width, new_height, channels, resized, new_width * channels);
    
    /*
    ref_time_simd = time_and_print(
        "Nearest neighbor CPU simd",
        cpu_nn_simd,
        image, resized,
        width, height,
        new_width, new_height,
        channels,
        data_size,
        ref_time_v1,   
        0
    );
    stbi_write_png("resized_cpu_nn_simd.png", new_width, new_height, channels, resized, new_width * channels);
    */

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
