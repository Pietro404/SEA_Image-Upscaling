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
        //if (src_y >= height) src_y = height - 1;

        for (int x = 0; x < new_width; x++) {
            int src_x = (int)(x * x_ratio);
            //if (src_x >= width) src_x = width - 1;

            int in_idx  = (src_y * width + src_x) * channels;
            int out_idx = (y * new_width + x) * channels;

            /*1. Questo introduce un overhead di controllo (incremento di c, confronto c < channels, salto condizionato) per ogni singolo pixel il costo è enorme.*/
            //for (int c = 0; c < channels; c++)
            //    output[out_idx + c] = input[in_idx + c];
            output[out_idx + 0] = input[in_idx + 0];
            output[out_idx + 1] = input[in_idx + 1];
            output[out_idx + 2] = input[in_idx + 2];

        }
    }
}
void cpu_nn_v2(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    // Usiamo il fixed-point a 16 bit (Shift di 16) per non convertire in float
    // Questo permette di calcolare gli indici con semplici operazioni intere
    const int FP_SHIFT = 16;

    int x_ratio = (width << FP_SHIFT) / new_width;
    int y_ratio = (height << FP_SHIFT) / new_height;

    // 1. LUT per gli indici X: rimuove il calcolo FP dal loop parallelo
    int* x_indices = (int*)_mm_malloc(new_width * sizeof(int), 16);
    for (int x = 0; x < new_width; x++) {
        x_indices[x] = ((x * x_ratio) >> FP_SHIFT) * channels;
    }

    for (int y = 0; y < new_height; y++) {
        int src_y = (y * y_ratio) >> FP_SHIFT;
        //if (src_y >= height) src_y = height - 1;
        unsigned char* src_row = &input[src_y * width * channels];
        unsigned char* dst_row = &output[y * new_width * channels];

        for (int x = 0; x < new_width; x++) {
            int in_x_idx = x_indices[x];

            //for (int c = 0; c < channels; c++)
            //    output[out_idx + c] = input[in_idx + c];
            int out_idx = x * channels;
            dst_row[out_idx + 0] = src_row[in_x_idx + 0];
            dst_row[out_idx + 1] = src_row[in_x_idx + 1];
            dst_row[out_idx + 2] = src_row[in_x_idx + 2];
        }
    }
    _mm_free(x_indices);
}

void cpu_nn_omp(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    // Usiamo il fixed-point a 16 bit (Shift di 16) per non convertire in float
    // Questo permette di calcolare gli indici con semplici operazioni intere
    const int FP_SHIFT = 16;

    int x_ratio = (width << FP_SHIFT) / new_width;
    int y_ratio = (height << FP_SHIFT) / new_height;

    // 1. LUT per gli indici X: rimuove il calcolo FP dal loop parallelo
    int* x_indices = (int*)_mm_malloc(new_width * sizeof(int), 16);
    for (int x = 0; x < new_width; x++) {
        x_indices[x] = ((x * x_ratio) >> FP_SHIFT) * channels;
    }
    #pragma omp parallel for
    for (int y = 0; y < new_height; y++) {
        int src_y = (y * y_ratio) >> FP_SHIFT;
        //if (src_y >= height) src_y = height - 1;
        // putnatori all'inizio della riga concorrente per input e output
        //usa la cache L1 invece di ram
        unsigned char* src_row = &input[src_y * width * channels];
        unsigned char* dst_row = &output[y * new_width * channels];

        for (int x = 0; x < new_width; x++) {
            int in_x_idx = x_indices[x];

            //for (int c = 0; c < channels; c++)
            //    output[out_idx + c] = input[in_idx + c];
            int out_idx = x * channels;
            dst_row[out_idx + 0] = src_row[in_x_idx + 0];
            dst_row[out_idx + 1] = src_row[in_x_idx + 1];
            dst_row[out_idx + 2] = src_row[in_x_idx + 2];
        }
    }
    _mm_free(x_indices);
}

#include <smmintrin.h> // Per _mm_shuffle_epi8. altrimenti con epi32: Ti servirebbero almeno 3-4 istruzioni (shuffle + shift + and + or)
void cpu_nn_sse2(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    if (channels != 3) return; // Supporto solo RGB

    const int FP_SHIFT = 16;
    int x_ratio_fp = (width << FP_SHIFT) / new_width;
    int y_ratio_fp = (height << FP_SHIFT) / new_height;

    // 1. UTILIZZO DI _mm_store_si128 PER INIZIALIZZARE LA LUT
    // Allocazione allineata a 16 byte obbligatoria per _mm_store_si128
    int* x_indices = (int*)_mm_malloc(new_width * sizeof(int), 16);
    
    // Possiamo vettorizzare anche la creazione della LUT se volessimo, 
    for (int x = 0; x < new_width; x++) {
        x_indices[x] = ((x * x_ratio_fp) >> FP_SHIFT) * 3;
    }

    // 2. UTILIZZO DI _mm_load_si128 PER CARICARE LA MASCHERA
    // Definiamo la maschera in memoria allineata a 16 byte
    alignas(16) const int8_t shuffle_mask_data[16] = {
        0, 1, 2,      // Pixel 0 (RGB)
        4, 5, 6,      // Pixel 1 (RGB)
        8, 9, 10,     // Pixel 2 (RGB)
        12, 13, 14,   // Pixel 3 (RGB)
        -1, -1, -1, -1 // Byte vuoti
    };

    // Carichiamo la maschera in un registro usando l'istruzione richiesta
    __m128i mask = _mm_load_si128((const __m128i*)shuffle_mask_data);

    for (int y = 0; y < new_height; y++) {
        // Calcolo riga sorgente
        int src_y = (y * y_ratio_fp) >> FP_SHIFT;
        if (src_y >= height) src_y = height - 1;

        unsigned char* src_row = &input[src_y * width * 3];
        unsigned char* dst_row = &output[y * new_width * 3];

        _mm_prefetch((const char*)src_row, _MM_HINT_T0);
        _mm_prefetch((const char*)(src_row + 64), _MM_HINT_T0);

        int x = 0;
        // Elaboriamo 16 pixel alla volta (48 byte di output)
        // 48 byte si scrivono perfettamente con 3 store da 16 byte (128 bit)
        for (; x <= new_width - 16; x += 16) {
            
            // A. FETCH INDICI (Usiamo _mm_load_si128 sulla LUT)
            // Carichiamo 4 indici alla volta dalla LUT allineata.
            // Nota: Anche se li carichiamo in SIMD, dobbiamo estrarli per usarli come indirizzi.
            __m128i v_idx0 = _mm_load_si128((__m128i*)&x_indices[x]);      // Indici 0-3
            __m128i v_idx1 = _mm_load_si128((__m128i*)&x_indices[x + 4]);  // Indici 4-7
            __m128i v_idx2 = _mm_load_si128((__m128i*)&x_indices[x + 8]);  // Indici 8-11
            __m128i v_idx3 = _mm_load_si128((__m128i*)&x_indices[x + 12]); // Indici 12-15

            // 1. CARICAMENTO PIXEL (4 vettori da 4 pixel l'uno)
            __m128i v0 = _mm_set_epi32(
                    *(uint32_t*)&src_row[_mm_extract_epi32(v_idx0, 3)], 
                    *(uint32_t*)&src_row[_mm_extract_epi32(v_idx0, 2)], 
                    *(uint32_t*)&src_row[_mm_extract_epi32(v_idx0, 1)], 
                    *(uint32_t*)&src_row[_mm_cvtsi128_si32(v_idx0)]
                );
            __m128i v1 = _mm_set_epi32(
                    *(uint32_t*)&src_row[_mm_extract_epi32(v_idx1, 3)], 
                    *(uint32_t*)&src_row[_mm_extract_epi32(v_idx1, 2)], 
                    *(uint32_t*)&src_row[_mm_extract_epi32(v_idx1, 1)], 
                    *(uint32_t*)&src_row[_mm_cvtsi128_si32(v_idx1)]
                );
            __m128i v2 = _mm_set_epi32(
                    *(uint32_t*)&src_row[_mm_extract_epi32(v_idx2, 3)], 
                    *(uint32_t*)&src_row[_mm_extract_epi32(v_idx2, 2)], 
                    *(uint32_t*)&src_row[_mm_extract_epi32(v_idx2, 1)], 
                    *(uint32_t*)&src_row[_mm_cvtsi128_si32(v_idx2)]
                );
            __m128i v3 = _mm_set_epi32(
                    *(uint32_t*)&src_row[_mm_extract_epi32(v_idx3, 3)], 
                    *(uint32_t*)&src_row[_mm_extract_epi32(v_idx3, 2)], 
                    *(uint32_t*)&src_row[_mm_extract_epi32(v_idx3, 1)], 
                    *(uint32_t*)&src_row[_mm_cvtsi128_si32(v_idx3)]
                );

            // 2. PACKING (Shuffle dei byte)
            v0 = _mm_shuffle_epi8(v0, mask);
            v1 = _mm_shuffle_epi8(v1, mask);
            v2 = _mm_shuffle_epi8(v2, mask);
            v3 = _mm_shuffle_epi8(v3, mask);

            // 3. STORE (Scrittura in memoria)
            // Prepariamo i 3 registri finali da 128 bit
            
            // Blocco 1 (Byte 0-15) tutto v0 (12 byte) + i primi 4 byte di v1
            __m128i s1 = _mm_or_si128(v0, _mm_bslli_si128(v1, 12));
            
            // Blocco 2 (Byte 16-31) v1 (8 byte) + v2 (8 byte)
            __m128i s2 = _mm_or_si128(_mm_bsrli_si128(v1, 4), _mm_bslli_si128(v2, 8));
            
            // Blocco 3 (Byte 32-47) i primi 4 byte di v2 + tutto v3 (12 byte)
            __m128i s3 = _mm_or_si128(_mm_bsrli_si128(v2, 8), _mm_bslli_si128(v3, 4));

            // 4. SCRITTURA CON _mm_store_si128
            //if (row_aligned) {
                // VELOCE: Scrittura allineata diretta (crash se l'indirizzo non è multiplo di 16)
                _mm_store_si128((__m128i*)(dst_row + x * channels + 0 ), s1);
                _mm_store_si128((__m128i*)(dst_row + x * channels + 16), s2);
                _mm_store_si128((__m128i*)(dst_row + x * channels + 32), s3);
            //} else {
            //    // SICURO: Scrittura non allineata (leggermente più lenta)
            //    _mm_storeu_si128((__m128i*)(dst_row + x * 3), s1);
            //    _mm_storeu_si128((__m128i*)(dst_row + x * 3 + 16), s2);
            //    _mm_storeu_si128((__m128i*)(dst_row + x * 3 + 32), s3);
            //}
        }

        // Tail loop (gestione residui scalari)
        for (; x < new_width; x++) {
            int off = x_indices[x]; // Qui leggiamo scalare, inutile caricare un vettore per 1 elemento
            dst_row[x * 3 + 0] = src_row[off + 0];
            dst_row[x * 3 + 1] = src_row[off + 1];
            dst_row[x * 3 + 2] = src_row[off + 2];
        }
    }
    _mm_free(x_indices);
}

void cpu_nn_sse2_v2(
    unsigned char* input,
    unsigned char* output,
    int width, int height,
    int new_width, int new_height,
    int channels
) {
    if (channels != 3) return;

    const int FP_SHIFT = 16;
    int x_ratio = (width << FP_SHIFT) / new_width;
    int y_ratio = (height << FP_SHIFT) / new_height;

    // LUT pre-calcolata (allineata a 16 byte )
    int* x_indices = (int*)_mm_malloc(new_width * sizeof(int), 16);
    for (int x = 0; x < new_width; x++) {
        x_indices[x] = ((x * x_ratio) >> FP_SHIFT) * channels;
    }
    // LOOP PRINCIPALE
    for (int y = 0; y < new_height; y++) {
        // Calcolo coordinata Y sorgente
        int src_y = (y * y_ratio) >> FP_SHIFT;
        //if (src_y >= height) src_y = height - 1;
        // Puntatori all'inizio della riga sorgente e destinazione
        unsigned char* src_row = &input[src_y * width * channels];
        unsigned char* dst_row = &output[y * new_width * channels];

        // Prefetch della riga sorgente per minimizzare i cache miss x2 (64 byte per cache line)
        _mm_prefetch((const char*)src_row, _MM_HINT_T0);
        _mm_prefetch((const char*)(src_row + 64), _MM_HINT_T0);

        int x = 0;
        // LOOP MASTER: 8 Pixel alla volta (24 byte scritti via 3x uint64_t)
        for (; x <= new_width - 8; x += 8) {
            // 1. LETTURA INDICI E PREFETCH DATI
            // Carichiamo gli indici dalla LUT in cache L1
            const int idx0 = x_indices[x];
            const int idx1 = x_indices[x+1];
            const int idx2 = x_indices[x+2];
            const int idx3 = x_indices[x+3];
            const int idx4 = x_indices[x+4];
            const int idx5 = x_indices[x+5];
            const int idx6 = x_indices[x+6];
            const int idx7 = x_indices[x+7];

            // 2. LETTURA PIXEL (Parallel Load)
            // Usiamo uint32_t per leggere 4 byte (RGB + 1 byte di scarto)
            // Il 4° byte è "spazzatura" che elimineremo con le maschere (AND).
            uint32_t p0 = *(uint32_t*)&src_row[idx0];
            uint32_t p1 = *(uint32_t*)&src_row[idx1];
            uint32_t p2 = *(uint32_t*)&src_row[idx2];
            uint32_t p3 = *(uint32_t*)&src_row[idx3];
            uint32_t p4 = *(uint32_t*)&src_row[idx4];
            uint32_t p5 = *(uint32_t*)&src_row[idx5];
            uint32_t p6 = *(uint32_t*)&src_row[idx6];
            uint32_t p7 = *(uint32_t*)&src_row[idx7];

            // PACKING 8 PIXEL in 3x 64-bit (24 Byte totali)
            // Store 1: P0(rgb), P1(rgb), P2(rg)
            uint64_t s0 = (uint64_t)(p0 & 0xFFFFFF) | 
                          ((uint64_t)(p1 & 0xFFFFFF) << 24) | 
                          ((uint64_t)(p2 & 0xFFFF) << 48);

            // Store 2: P2(b), P3(rgb), P4(rgb), P5(r)
            uint64_t s1 = (uint64_t)((p2 >> 16) & 0xFF) | 
                          ((uint64_t)(p3 & 0xFFFFFF) << 8) | 
                          ((uint64_t)(p4 & 0xFFFFFF) << 32) |
                          ((uint64_t)(p5 & 0xFF) << 56);

            // Store 3: P5(gb), P6(rgb), P7(rgb)
            uint64_t s2 = (uint64_t)((p5 >> 8) & 0xFFFF) | 
                          ((uint64_t)(p6 & 0xFFFFFF) << 16) | 
                          ((uint64_t)(p7 & 0xFFFFFF) << 40);

            // 3. SCRITTURA BULK
            // Scriviamo i 24 byte (8 pixel) in 3 operazioni da 8 byte ciascuna
            uint64_t* d = (uint64_t*)&dst_row[x * 3];
            d[0] = s0;
            d[1] = s1;
            d[2] = s2;
        }

        // Tail loop per i rimasugli
        // Se la larghezza non è multipla di 8, finiamo i pixel rimanenti uno ad uno.
        for (; x < new_width; x++) {
            int off = x_indices[x];
            dst_row[x * channels]     = src_row[off];
            dst_row[x * channels + 1] = src_row[off + 1];
            dst_row[x * channels + 2] = src_row[off + 2];
        }
    }
    // Liberiamo la memoria della LUT
    _mm_free(x_indices);
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
    long ref_time_sse2;
    long ref_time_sse2_v2;
    long ref_time_sse2_v4;

    ref_time_v1 = time_and_print(
        "Nearest neighbor CPU v1\t",
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
        "Nearest neighbor CPU v2\t",
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
        "Nearest neighbor CPU omp\t",
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
    
    ref_time_sse2 = time_and_print(
        "Nearest neighbor CPU sse2\t",
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

    ref_time_sse2_v2 = time_and_print(
        "Nearest neighbor CPU sse2_2\t",
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
    
    stbi_image_free(image);
    free(resized);

    printf("Upscaling RGB completato!\n");
    return 0;
}
