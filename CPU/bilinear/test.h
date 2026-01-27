#ifndef TEST_H
#define TEST_H

#include <cstddef>

// Firma del kernel
using kernel_fn = void (*)(
    unsigned char*, unsigned char*,
    int, int, int, int, int
    );

// Funzione di benchmark
long time_and_print(
    const char* name,
    kernel_fn kernel,
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int new_width,
    int new_height,
    int channels,
    size_t data_size,
    long ref_time,
    long computation_units
);

#endif // TIME_H