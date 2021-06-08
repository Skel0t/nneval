#include <iostream>

#include "interface.h"
#include "image.h"
#include "nn.h"

void sparseMult();
void imageTest();
void create_conv();

int main() {
    // imageTest();
    auto kernel1 = read_in_weigths2("/home/woshi/Documents/nneval/network/conv1.txt", 3, 32, 5);
    auto kernel2 = read_in_weigths2("/home/woshi/Documents/nneval/network/conv2.txt", 32, 64, 3);
    auto kernel3 = read_in_weigths2("/home/woshi/Documents/nneval/network/conv3.txt", 64, 64, 3);
    auto upkernel1 = read_in_weigths2("/home/woshi/Documents/nneval/network/upconv1.txt", 64, 32, 5);
    auto kernel4 = read_in_weigths2("/home/woshi/Documents/nneval/network/conv4.txt", 32, 32, 3);
    auto kernel5 = read_in_weigths2("/home/woshi/Documents/nneval/network/conv5.txt", 32, 32, 3);
    auto kernel6 = read_in_weigths2("/home/woshi/Documents/nneval/network/conv6.txt", 32,  3, 3);

    auto c1Bias = read_in_biases("/home/woshi/Documents/nneval/network/c1bias.txt", 32);
    auto c2Bias = read_in_biases("/home/woshi/Documents/nneval/network/c2bias.txt", 64);
    auto c3Bias = read_in_biases("/home/woshi/Documents/nneval/network/c3bias.txt", 64);
    auto upc1Bias = read_in_biases("/home/woshi/Documents/nneval/network/uc1bias.txt", 32);
    auto c4Bias = read_in_biases("/home/woshi/Documents/nneval/network/c4bias.txt", 32);
    auto c5Bias = read_in_biases("/home/woshi/Documents/nneval/network/c5bias.txt", 32);

    std::cout << (*kernel1)[0]->data()[2] << std::endl;

    ImageRgba32 img;
    load_png(FilePath("/home/woshi/Documents/nneval/src/mitchell.png"), img);

    sres(kernel1, kernel2, kernel3, upkernel1, kernel4, kernel5, kernel6, c1Bias, c2Bias, c3Bias, upc1Bias, c4Bias, c5Bias, c5Bias, &img.pixels);

    return 0;
}

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