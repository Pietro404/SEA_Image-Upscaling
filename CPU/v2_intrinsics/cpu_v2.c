#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <emmintrin.h>
#include <stdio.h>
#include <stdlib.h>
 
// Nearest Neighbor RGB 
// Nearest Neighbor presenta accessi regolari e indipendenti, risultando particolarmente adatto alla vettorizzazione SIMD.
void nearest_neighbor_sse2(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int new_width,
    int new_height,
    int channels
) {
    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    for (int y = 0; y < new_height; y++) {
        int src_y = (int)(y * y_ratio);
        if (src_y >= height) src_y = height - 1;

        for (int x = 0; x <= new_width - 4; x += 4) {

            int sx[4];
            for (int i = 0; i < 4; i++) {
                sx[i] = (int)((x + i) * x_ratio);
                if (sx[i] >= width) sx[i] = width - 1;
            }

            for (int c = 0; c < channels; c++) {
                __m128i pix = _mm_setr_epi32(
                    input[(src_y * width + sx[0]) * channels + c],
                    input[(src_y * width + sx[1]) * channels + c],
                    input[(src_y * width + sx[2]) * channels + c],
                    input[(src_y * width + sx[3]) * channels + c]
                );

                int tmp[4];
                _mm_storeu_si128((__m128i*)tmp, pix);

                for (int i = 0; i < 4; i++) {
                    output[((y * new_width + x + i) * channels) + c] =
                        (unsigned char)tmp[i];
                }
            }
        }
    }
}

//bilinear interpolation RPG
//La vettorizzazione SSE2 è stata applicata alle operazioni aritmetiche, 
// mentre gli accessi ai pixel sono rimasti scalari per evitare overhead dovuti a gather manuali.
void bilinear_interpolation_sse2(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int new_width,
    int new_height,
    int channels
) {
    float x_ratio = (float)(width - 1) / new_width;
    float y_ratio = (float)(height - 1) / new_height;

    for (int y = 0; y < new_height; y++) {
        float gy = y * y_ratio;
        int y0 = (int)gy;
        int y1 = (y0 + 1 < height) ? y0 + 1 : y0;
        float dy = gy - y0;

        __m128 dyv = _mm_set1_ps(dy);
        __m128 one = _mm_set1_ps(1.0f);

        for (int x = 0; x <= new_width - 4; x += 4) {

            float dx[4];
            int x0[4], x1[4];

            for (int i = 0; i < 4; i++) {
                float gx = (x + i) * x_ratio;
                x0[i] = (int)gx;
                x1[i] = (x0[i] + 1 < width) ? x0[i] + 1 : x0[i];
                dx[i] = gx - x0[i];
            }

            __m128 dxv = _mm_loadu_ps(dx);

            for (int c = 0; c < channels; c++) {
                float p00[4], p10[4], p01[4], p11[4];

                for (int i = 0; i < 4; i++) {
                    p00[i] = input[(y0 * width + x0[i]) * channels + c];
                    p10[i] = input[(y0 * width + x1[i]) * channels + c];
                    p01[i] = input[(y1 * width + x0[i]) * channels + c];
                    p11[i] = input[(y1 * width + x1[i]) * channels + c];
                }

                __m128 P00 = _mm_loadu_ps(p00);
                __m128 P10 = _mm_loadu_ps(p10);
                __m128 P01 = _mm_loadu_ps(p01);
                __m128 P11 = _mm_loadu_ps(p11);

                __m128 val =
                    _mm_add_ps(
                        _mm_add_ps(
                            _mm_mul_ps(P00, _mm_mul_ps(_mm_sub_ps(one, dxv), _mm_sub_ps(one, dyv))),
                            _mm_mul_ps(P10, _mm_mul_ps(dxv, _mm_sub_ps(one, dyv)))
                        ),
                        _mm_add_ps(
                            _mm_mul_ps(P01, _mm_mul_ps(_mm_sub_ps(one, dxv), dyv)),
                            _mm_mul_ps(P11, _mm_mul_ps(dxv, dyv))
                        )
                    );

                float out[4];
                _mm_storeu_ps(out, val);

                for (int i = 0; i < 4; i++) {
                    output[((y * new_width + x + i) * channels) + c] =
                        (unsigned char)out[i];
                }
            }
        }
    }
}

//L’interpolazione bicubica utilizza una finestra 4×4 e accessi non contigui alla memoria. 
// In SSE2 l’assenza di istruzioni gather rende la vettorizzazione inefficiente; 
// pertanto l’algoritmo è stato mantenuto scalare.

/*void bicubic_interpolation(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int new_width,
    int new_height,
    int channels
) {
    float x_ratio = (float)width / new_width;
    float y_ratio = (float)height / new_height;

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {

            float gx = x * x_ratio;
            float gy = y * y_ratio;

            int x_int = (int)gx;
            int y_int = (int)gy;

            float dx = gx - x_int;
            float dy = gy - y_int;

            for (int c = 0; c < channels; c++) {
                float value = 0.0f;

                for (int m = -1; m <= 2; m++) {
                    int yy = y_int + m;
                    if (yy < 0) yy = 0;
                    if (yy >= height) yy = height - 1;

                    float wy = cubic(m - dy);

                    for (int n = -1; n <= 2; n++) {
                        int xx = x_int + n;
                        if (xx < 0) xx = 0;
                        if (xx >= width) xx = width - 1;

                        float wx = cubic(n - dx);

                        int idx = (yy * width + xx) * channels + c;
                        value += input[idx] * wx * wy;
                    }
                }

                if (value < 0) value = 0;
                if (value > 255) value = 255;

                output[(y * new_width + x) * channels + c] =
                    (unsigned char)(value);
            }
        }
    }
}*/

int main() {
    int width, height, channels;

    // Carichiamo l'immagine RGB
    unsigned char* image = stbi_load(
        "../mario.png",
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
        malloc(new_width * new_height * channels);

    if (!resized) {
        printf("Errore allocazione memoria\n");
        stbi_image_free(image);
        return 1;
    }

    nearest_neighbor_sse2(
        image,
        resized,
        width,
        height,
        new_width,
        new_height,
        channels
    );
    
    stbi_write_png(
        "mario_cpu_v2_nn.png",
        new_width,
        new_height,
        channels,
        resized,
        new_width * channels
    );
     
    /*bilinear_interpolation_sse2(
        image,
        resized,
        width,
        height,
        new_width,
        new_height,
        channels
    );

    stbi_write_png(
        "mario_cpu_v2_bilinear.png",
        new_width,
        new_height,
        channels,
        resized,
        new_width * channels
    );*/

    /*bicubic_interpolation(
        image,
        resized,
        width,
        height,
        new_width,
        new_height,
        channels
    );

    stbi_write_png(
        "images_cpu_v2_bicubic.png",
        new_width,
        new_height,
        channels,
        resized,
        new_width * channels
    );*/

    stbi_image_free(image);
    free(resized);

    printf("Upscaling RGB completato!\n");
    return 0;
}
