#include <iostream>
#include <fstream>

#include "interface.h"
#include "nn.h"

float* read_in_weigths(std::string path, int in_channels, int out_channels, int ksize) {
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Couldn't open" << std::endl;
        return nullptr;
    } else {
        float* ptr = (float*) malloc(sizeof(float) * ksize * ksize * in_channels * out_channels);

        for (int i = 0; i < out_channels; i++) {
            int k_nr = i * in_channels * ksize * ksize;
            for (int j = 0; j < in_channels; j++) {
                for (int y = 0; y < ksize; y++) {
                    int k_row = y * ksize * in_channels;
                    for (int x = 0; x < ksize; x++) {
                        f >> ptr[k_nr + k_row + x * in_channels + j];
                    }
                }
            }
        }
        float x;
        return ptr;
    }
}

anydsl::Array<anydsl::Array<float>*>* read_in_weigths2(std::string path, int in_channels, int out_channels, int ksize) {
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Couldn't open" << std::endl;
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
        float x;
        return outer_ptr;
    }
}

float* read_in_biases(std::string path, int out_channels) {
    std::fstream f;
    f.open(path, std::ios::in);
    if (!f) {
        std::cout << "Couldn't open" << std::endl;
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