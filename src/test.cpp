#include <iostream>

#include "interface.h"
#include "image.h"
#include "nn.h"

void imageTest();
void im2col(std::string path);
void superres_conv_mat(std::string path);
void superres_im2col(std::string path);

int main() {
    superres_im2col("/home/woshi/Documents/nneval/src/orig.png");
    return 0;
}

void im2col(std::string path) {
    const int memsize1 = 5 * 5 *  3 * 32;
    const int memsize2 = 3 * 3 * 32 * 64;
    const int memsize3 = 3 * 3 * 64 * 64;
    const int memsize4 = 5 * 5 * 64 * 32;
    const int memsize5 = 3 * 3 * 32 * 32;
    const int memsize6 = 3 * 3 * 32 * 32;
    const int memsize7 = 3 * 3 * 32 *  3;

    // Buffer for all convolution weights
    anydsl::Array<float> weights(sizeof(float) * (memsize1 + memsize2 + memsize3 + memsize4 + memsize5 + memsize6 + memsize7));

    int offset = 0;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv1.txt",    3, 32, 5);
    offset += memsize1;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv2.txt",   32, 64, 3);
    offset += memsize2;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv3.txt",   64, 64, 3);
    offset += memsize3;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/upconv1.txt", 64, 32, 5);
    offset += memsize4;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv4.txt",   32, 32, 3);
    offset += memsize5;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv5.txt",   32, 32, 3);
    offset += memsize6;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv6.txt",   32,  3, 3);

    ImageRgba32 img;
    load_png(FilePath(path), img);
    int width  = img.width;
    int height = img.height;
    anydsl::Array<uint8_t> result(sizeof(uint8_t) * (width) * (height) * 3);

    im2col_test(&img.pixels, width, height, &weights, &result);

    save_png_pointer(FilePath("out_im2col.png"), img.pixels.data(), width, height);

    // Free all allocated memory
    weights.release();
    img.pixels.release();
    result.release();
}

void superres_im2col(std::string path) {
    // Necessary memory: "sizeof(float) * ksize * ksize * in_channels * out_channels" for each convolution
    const int memsize1 = 5 * 5 *  3 * 32;
    const int memsize2 = 3 * 3 * 32 * 64;
    const int memsize3 = 3 * 3 * 64 * 64;
    const int memsize4 = 5 * 5 * 64 * 32;
    const int memsize5 = 3 * 3 * 32 * 32;
    const int memsize6 = 3 * 3 * 32 * 32;
    const int memsize7 = 3 * 3 * 32 *  3;

    const int memsize_weights = memsize1 + memsize2 + memsize3 + memsize4 + memsize5 + memsize6 + memsize7;
    const int memsize_biases  = 32 + 64 + 64 + 32 + 32 + 32;

    // Buffer for all convolution weights
    anydsl::Array<float> weights(sizeof(float) * (memsize_weights));

    // Buffer for all convolution biases, one for each out_channel
    float* biases = (float*) malloc(sizeof(float) * (memsize_biases));

    int offset = 0;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv1.txt",    3, 32, 5);
    offset += memsize1;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv2.txt",   32, 64, 3);
    offset += memsize2;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv3.txt",   64, 64, 3);
    offset += memsize3;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/upconv1.txt", 64, 32, 5);
    offset += memsize4;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv4.txt",   32, 32, 3);
    offset += memsize5;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv5.txt",   32, 32, 3);
    offset += memsize6;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv6.txt",   32,  3, 3);

    read_in_biases(biases, 0                     , "/home/woshi/Documents/superres/src/network/c1bias.txt",  32);
    read_in_biases(biases, 32                    , "/home/woshi/Documents/superres/src/network/c2bias.txt",  64);
    read_in_biases(biases, 32 + 64               , "/home/woshi/Documents/superres/src/network/c3bias.txt",  64);
    read_in_biases(biases, 32 + 64 + 64          , "/home/woshi/Documents/superres/src/network/uc1bias.txt", 32);
    read_in_biases(biases, 32 + 64 + 64 + 32     , "/home/woshi/Documents/superres/src/network/c4bias.txt",  32);
    read_in_biases(biases, 32 + 64 + 64 + 32 + 32, "/home/woshi/Documents/superres/src/network/c5bias.txt",  32);
    // conv 6 doesnt have a bias

    ImageRgba32 img;
    load_png(FilePath(path), img);
    int width  = img.width;
    int height = img.height;
    anydsl::Array<uint8_t> result(sizeof(uint8_t) * (2 * width) * (2 * height) * 3);

    std::cout << "Memory in cpp:\t\t" << 4 * (memsize_biases + memsize_weights) + sizeof(uint8_t) * (2 * width) * (2 * height) * 3 + width * height * 3 << std::endl;

    sres_im2col(&img.pixels, width, height, &result, &weights, biases);

    save_png_pointer(FilePath("out_im2col.png"), result.data(), width * 2, height * 2);

    // Free all allocated memory
    weights.release();
    img.pixels.release();
    result.release();
    free(biases);
}

void superres_conv_mat(std::string path) {
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
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv1.txt",    3, 32, 5);
    offset += memsize1;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv2.txt",   32, 64, 3);
    offset += memsize2;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv3.txt",   64, 64, 3);
    offset += memsize3;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/upconv1.txt", 64, 32, 5);
    offset += memsize4;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv4.txt",   32, 32, 3);
    offset += memsize5;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv5.txt",   32, 32, 3);
    offset += memsize6;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/superres/src/network/conv6.txt",   32,  3, 3);

    read_in_biases(biases, 0                     , "/home/woshi/Documents/superres/src/network/c1bias.txt",  32);
    read_in_biases(biases, 32                    , "/home/woshi/Documents/superres/src/network/c2bias.txt",  64);
    read_in_biases(biases, 32 + 64               , "/home/woshi/Documents/superres/src/network/c3bias.txt",  64);
    read_in_biases(biases, 32 + 64 + 64          , "/home/woshi/Documents/superres/src/network/uc1bias.txt", 32);
    read_in_biases(biases, 32 + 64 + 64 + 32     , "/home/woshi/Documents/superres/src/network/c4bias.txt",  32);
    read_in_biases(biases, 32 + 64 + 64 + 32 + 32, "/home/woshi/Documents/superres/src/network/c5bias.txt",  32);
    // conv 6 doesnt have a bias

    ImageRgba32 img;
    load_png(FilePath(path), img);
    int width  = img.width;
    int height = img.height;
    anydsl::Array<uint8_t> result(sizeof(uint8_t) * (2 * width) * (2 * height) * 3);

    sres_conv_mat(&img.pixels, width, height, &result, &weights, biases);

    save_png_pointer(FilePath("out_convmat.png"), result.data(), width * 2, height * 2);

    // Free all allocated memory
    weights.release();
    img.pixels.release();
    result.release();
    free(biases);
}

void imageTest() {
    float kernel1[] = { .0f,    .0f,    .0f,     .0f,     .0f,     .0f,    .0f,    .0f,    .0f, // R
                        .0f,    .0f,    .0f,     .0f,     1.f,     .0f,    .0f,    .0f,    .0f, // G
                        .0f,    .0f,    .0f,     .0f,     .0f,     .0f,    .0f,    .0f,    .0f};// B
    anydsl::Array<float> dsl_kernel1(sizeof(float) * 27);
    anydsl_copy(0, kernel1, 0, 0, dsl_kernel1.data(), 0, 27 * sizeof(float));

    ImageRgba32 img;
    load_png(FilePath("/home/woshi/Documents/nneval/src/mitchell.png"), img);

    anydsl::Array<uint8_t> result(sizeof(uint8_t) * 4 * 1920 * 1110 * 3);

    image_kernel_test(&dsl_kernel1, &img.pixels, &result);

    save_png_pointer(FilePath("out.png"), result.data(), 1920 * 2, 2 * 1110);

    img.pixels.release();
    dsl_kernel1.release();
    result.release();
}
