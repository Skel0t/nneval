#include <iostream>

#include "interface.h"
#include "image.h"
#include "nn.h"
/*
void sparseMult();
void imageTest();
void create_conv();`*/
void superres(std::string path, int width, int height);

int main() {
    superres("/home/woshi/Documents/nneval/src/65010.png", 240, 160);
    return 0;
}

void superres(std::string path, int width, int height) {
    // Necessary memory: "sizeof(float) * ksize * ksize * in_channels * out_channels" for each convolution
    const int memsize1 = 5 * 5 *  3 * 32;
    const int memsize2 = 3 * 3 * 32 * 64;
    const int memsize3 = 3 * 3 * 64 * 64;
    const int memsize4 = 5 * 5 * 64 * 32;
    const int memsize5 = 3 * 3 * 32 * 32;
    const int memsize6 = 3 * 3 * 32 * 32;
    const int memsize7 = 3 * 3 * 32 *  3;

    // Buffer for all convolution weights
    anydsl::Array<float> weights(sizeof(float) * (memsize1 + memsize2 + memsize3 + memsize4 + memsize5 + memsize6 + memsize7));

    // Buffer for all convolution biases, one for each out_channel
    float* biases = (float*) malloc(sizeof(float) * (32 + 64 + 64 + 32 + 32 + 32));

    int offset = 0;
    read_in_weigths(&weights, offset, "/home/woshi/Documents/superres/src/network/conv1.txt",    3, 32, 5);
    offset += memsize1;
    read_in_weigths(&weights, offset, "/home/woshi/Documents/superres/src/network/conv2.txt",   32, 64, 3);
    offset += memsize2;
    read_in_weigths(&weights, offset, "/home/woshi/Documents/superres/src/network/conv3.txt",   64, 64, 3);
    offset += memsize3;
    read_in_weigths(&weights, offset, "/home/woshi/Documents/superres/src/network/upconv1.txt", 64, 32, 5);
    offset += memsize4;
    read_in_weigths(&weights, offset, "/home/woshi/Documents/superres/src/network/conv4.txt",   32, 32, 3);
    offset += memsize5;
    read_in_weigths(&weights, offset, "/home/woshi/Documents/superres/src/network/conv5.txt",   32, 32, 3);
    offset += memsize6;
    read_in_weigths(&weights, offset, "/home/woshi/Documents/superres/src/network/conv6.txt",   32,  3, 3);

    read_in_biases(biases, 0                     , "/home/woshi/Documents/superres/src/network/c1bias.txt",  32);
    read_in_biases(biases, 32                    , "/home/woshi/Documents/superres/src/network/c2bias.txt",  64);
    read_in_biases(biases, 32 + 64               , "/home/woshi/Documents/superres/src/network/c3bias.txt",  64);
    read_in_biases(biases, 32 + 64 + 64          , "/home/woshi/Documents/superres/src/network/uc1bias.txt", 32);
    read_in_biases(biases, 32 + 64 + 64 + 32     , "/home/woshi/Documents/superres/src/network/c4bias.txt",  32);
    read_in_biases(biases, 32 + 64 + 64 + 32 + 32, "/home/woshi/Documents/superres/src/network/c5bias.txt",  32);
    // conv 6 doesnt have a bias

    ImageRgba32 img;
    load_png(FilePath(path), img);
    anydsl::Array<uint8_t> result(sizeof(uint8_t) * (2 * width) * (2 * height) * 3);

    sres(&img.pixels, width, height, &result, &weights, biases);

    save_png_pointer(FilePath("out.png"), img.pixels.data(), width * 2, height * 2);

    // Free all allocated memory
    weights.release();
    img.pixels.release();
    result.release();
    free(biases);
}
/*
void create_conv() {
    float arr1[] = {    4.f,    3.f,    2.f,    1.f,
                        5.f,    6.f,    7.f,    8.f,
                        12.f,   11.f,   10.f,   9.f,
                        13.f,   14.f,   15.f,   16.f};
    float arr3[] = {    1.f,    1.f,   1.f,
                        1.f,    1.f,   1.f,
                        1.f,    1.f,   1.f};

    anydsl::Array<float> dslarr1(sizeof(float) * 16);
    anydsl::Array<float> dslarr3(sizeof(float) * 9);
    anydsl_copy(0, arr1, 0, 0, dslarr1.data(), 0, 16 * sizeof(float));
    anydsl_copy(0, arr3, 0, 0, dslarr3.data(), 0, 9 * sizeof(float));

    create_conv_test(&dslarr1, &dslarr3);
}

void sparseMult() {
    float arr1[] = {    2.f,   1.f,
                        2.f,   3.f};
    int arr2[] =   {    0,      1,
                        1,      2};

    float arr3[] = {    3.f,   2.f,   1.f,
                        4.f,   5.f,   6.f,
                        9.f,   8.f,   7.f};

    anydsl::Array<float> dslarr1(sizeof(float) * 9);
    anydsl::Array<int> dslarr2(sizeof(float) * 9);
    anydsl::Array<float> dslarr3(sizeof(float) * 9);
    anydsl_copy(0, arr1, 0, 0, dslarr1.data(), 0, 9 * sizeof(float));
    anydsl_copy(0, arr2, 0, 0, dslarr2.data(), 0, 9 * sizeof(int));
    anydsl_copy(0, arr3, 0, 0, dslarr3.data(), 0, 9 * sizeof(float));

    sparse_mult_test(&dslarr1, &dslarr2, &dslarr3);
}

void imageTest() {
    float kernel1[] = { .0f,    .0f,    .0f,     .0f,     .0f,     .0f,    .0f,    .0f,    .0f,
                        .0f,    .0f,    .0f,     1.f,     .0f,     .0f,    .0f,    .0f,    .0f,
                        .0f,    .0f,    .0f,     .0f,     .0f,     .0f,    .0f,    .0f,    .0f};
    float kernel2[] = { .0f,    .0f,    .0f,     .0f,     .0f,     .0f,    .0f,    .0f,    .0f,
                        .0f,    .0f,    .0f,     .0,      1.f,     .0f,    .0f,    .0f,    .0f,
                        .0f,    .0f,    .0f,     .0f,     .0f,     .0f,    .0f,    .0f,    .0f};
    float kernel3[] = { .0f,    .0f,    .0f,     .0f,     .0f,     .0f,    .0f,    .0f,    .0f,
                        .0f,    .0f,    .0f,     .0f,     .0f,     1.f,    .0f,    .0f,    .0f,
                        .0f,    .0f,    .0f,     .0f,     .0f,     .0f,    .0f,    .0f,    .0f};
    anydsl::Array<float> dsl_kernel1(sizeof(float) * 27);
    anydsl::Array<float> dsl_kernel2(sizeof(float) * 27);
    anydsl::Array<float> dsl_kernel3(sizeof(float) * 27);
    anydsl_copy(0, kernel1, 0, 0, dsl_kernel1.data(), 0, 27 * sizeof(float));
    anydsl_copy(0, kernel2, 0, 0, dsl_kernel2.data(), 0, 27 * sizeof(float));
    anydsl_copy(0, kernel3, 0, 0, dsl_kernel3.data(), 0, 27 * sizeof(float));

    ImageRgba32 img;
    load_png(FilePath("/home/woshi/Documents/nneval/src/mitchell.png"), img);

    uint8_t* ptr = image_kernel_test(&dsl_kernel1, &dsl_kernel2, &dsl_kernel3, &img.pixels);

    img.pixels.release();
    dsl_kernel1.release();
    dsl_kernel2.release();
    dsl_kernel3.release();

    save_png_pointer(FilePath("out.png"), ptr, 1920, 1110);
}
*/