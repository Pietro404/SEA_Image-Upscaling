#include "test.h"

#include <cstdio>
#include <chrono>
#include <string>


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
    long ref_time,   // baseline
    long computation_units
) {
    // 1. Usiamo i microsecondi per maggiore precisione (fondamentale per SSE/AVX)
    auto start = std::chrono::high_resolution_clock::now();

    kernel(input, output,
        width, height,
        new_width, new_height,
        channels);

    auto end = std::chrono::high_resolution_clock::now();

    long time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    if (time_us == 0) time_us = 1; // Previene divisioni per zero

    // 2. Calcolo corretto dei byte processati
    size_t input_bytes = (size_t)width * height * channels;
    size_t output_bytes = (size_t)new_width * new_height * channels;

    // Fattore di lettura in base all'algoritmo
    double read_multiplier = 1.0;
    std::string n(name);
    if (n.find("Bilinear") != std::string::npos) read_multiplier = 4.0;
    else if (n.find("Bicubic") != std::string::npos) read_multiplier = 16.0;

    // 3. Bandwidth Reale = (Letture Effettive + Scritture) / Tempo
    // Le letture totali sono output_pixels * read_multiplier
    double total_bytes = (double)output_bytes + ((double)output_bytes * read_multiplier);
    
    // Convertiamo microsecondi in secondi per il calcolo GB/s (1e-6 s)
    double time_s = time_us / 1e6;
    double throughput_gbs = (output_bytes / 1e9) / time_s;
    double bandwidth_gbs = (total_bytes / 1e9) / time_s;

    // 4. Stampa formattata
    printf(" -> %-22s ", name);
    printf("\t time: %7.2f ms", time_us / 1000.0); // Mostriamo ms con decimali
    printf("\t | Througput: %5.2f GB/s", throughput_gbs);
    printf("\t | Bandwidth: %5.2f GB/s", bandwidth_gbs);

    if (ref_time > 0) {
        double speedup = (double)ref_time / (double)time_us;
        printf("\t | Speedup: %5.2fx", speedup);

        if (computation_units > 1) {
            // Efficienza parallela (quanto bene scaliamo sui core)
            double efficiency = speedup / (double)computation_units;
            printf(" | comp_units: %d Eff: %4.2f",computation_units, efficiency);
        }
    }

    printf("\n");

    return time_us; // Restituiamo microsecondi per mantenere precisione nei test successivi
}