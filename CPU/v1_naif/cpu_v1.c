#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>

// Nearest Neighbor RGB
void nearest_neighbor(
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

            int src_x = (int)(x * x_ratio);
            int src_y = (int)(y * y_ratio);

            if (src_x >= width)  src_x = width - 1;
            if (src_y >= height) src_y = height - 1;

            for (int c = 0; c < channels; c++) {
                output[(y * new_width + x) * channels + c] =
                    input[(src_y * width + src_x) * channels + c];
            }
        }
    }
}

//bilinear interpolation RPG
void bilinear_interpolation(
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
        for (int x = 0; x < new_width; x++) {

            float gx = x * x_ratio;
            float gy = y * y_ratio;

            int x0 = (int)gx;
            int y0 = (int)gy;
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            if (x1 >= width)  x1 = width - 1;
            if (y1 >= height) y1 = height - 1;

            float dx = gx - x0;
            float dy = gy - y0;

            for (int c = 0; c < channels; c++) {

                int idx00 = (y0 * width + x0) * channels + c;
                int idx10 = (y0 * width + x1) * channels + c;
                int idx01 = (y1 * width + x0) * channels + c;
                int idx11 = (y1 * width + x1) * channels + c;

                float p00 = input[idx00];
                float p10 = input[idx10];
                float p01 = input[idx01];
                float p11 = input[idx11];

                float value =
                    p00 * (1 - dx) * (1 - dy) +
                    p10 * dx * (1 - dy) +
                    p01 * (1 - dx) * dy +
                    p11 * dx * dy;

                output[(y * new_width + x) * channels + c] =
                    (unsigned char)value;
            }
        }
    }
}

static float cubic(float x) {
    const float a = -0.5f; // Catmull-Rom
    if (x < 0) x = -x;

    if (x <= 1.0f)
        return (a + 2) * x * x * x - (a + 3) * x * x + 1;
    else if (x < 2.0f)
        return a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a;
    else
        return 0.0f;
}

void bicubic_interpolation(
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
}

int main() {
    int width, height, channels;

    // Carichiamo l'immagine RGB
    unsigned char* image = stbi_load(
        "images.jpg",
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

    /*nearest_neighbor(
        image,
        resized,
        width,
        height,
        new_width,
        new_height,
        channels
    );
    
    stbi_write_png(
        "mario_cpu_v1_nn.png",
        new_width,
        new_height,
        channels,
        resized,
        new_width * channels
    );*/
     
    /*bilinear_interpolation(
        image,
        resized,
        width,
        height,
        new_width,
        new_height,
        channels
    );

    stbi_write_png(
        "mario_cpu_v1_bilinear.png",
        new_width,
        new_height,
        channels,
        resized,
        new_width * channels
    );*/

    bicubic_interpolation(
        image,
        resized,
        width,
        height,
        new_width,
        new_height,
        channels
    );

    stbi_write_png(
        "images_cpu_v1_bicubic.png",
        new_width,
        new_height,
        channels,
        resized,
        new_width * channels
    );

    stbi_image_free(image);
    free(resized);

    printf("Upscaling RGB completato!\n");
    return 0;
}
