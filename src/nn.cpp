#include <iostream>
#include <fstream>

#include "interface.h"
#include "nn.h"

void read_in_weigths(anydsl::Array<float>* buffer, int offset, std::string path, int in_channels, int out_channels, int ksize) {
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Couldn't open " << path << std::endl;
    } else {
        for (int i = 0; i < out_channels; i++) {
            int k_nr = i * in_channels * ksize * ksize;
            for (int j = 0; j < in_channels; j++) {
                for (int y = 0; y < ksize; y++) {
                    int k_row = y * ksize * in_channels;
                    for (int x = 0; x < ksize; x++) {
                        f >> buffer->data()[offset + k_nr + k_row + x * in_channels + j];
                    }
                }
            }
        }
    }
}

anydsl::Array<anydsl::Array<float>*>* read_in_weigths2(std::string path, int in_channels, int out_channels, int ksize) {
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Couldn't open" << path << std::endl;
        return nullptr;
    } else {
        anydsl::Array<anydsl::Array<float>*>* outer_ptr = new anydsl::Array<anydsl::Array<float>*>(sizeof(anydsl::Array<float>*) * out_channels);

        for (int i = 0; i < out_channels; i++) {
            anydsl::Array<float>* ptr = new anydsl::Array<float>(sizeof(float) * ksize * ksize * in_channels);
            for (int j = 0; j < in_channels; j++) {
                for (int y = 0; y < ksize; y++) {
                    int k_row = y * ksize * in_channels;
                    for (int x = 0; x < ksize; x++) {
                        float n;
                        f >> n;
                        ptr->data()[(k_row + x * in_channels + j)] = n;
                    }
                }
            }
            (*outer_ptr)[i] = ptr;
        }
        return outer_ptr;
    }
}

void read_in_biases(float* buffer, int offset, std::string path, int out_channels) {
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Couldn't open" << path << std::endl;
    } else {
        for (int i = 0; i < out_channels; i++) {
            f >> buffer[offset + i];
        }
    }
}

float* read_in_biases2(std::string path, int out_channels) {
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Couldn't open" << path << std::endl;
        return nullptr;
    } else {
        float* ptr = (float*) malloc(sizeof(float) * out_channels);

        for (int i = 0; i < out_channels; i++) {
            f >> ptr[i];
        }
        float x;
        return ptr;
    }
}