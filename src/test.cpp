#include <iostream>
#include <chrono>

#include "interface.h"
#include "image.h"
#include "nn.h"
/*
void imageTest();
void im2col(std::string path);
void superres_conv_mat(std::string path);
void superres_im2col(std::string path);
void denoise(std::string path1, std::string path2, std::string path3);
*/
void bench();

int main() {
    bench();
    return 0;
}

void bench() {
    const int in_channels  = 64;
    const int out_channels = 32;
    const int ksize  = 5;
    const int width  = 960;
    const int height = 540;
    const int memsize_weights = ksize * ksize * in_channels * out_channels;
    const int memsize_biases  = out_channels;
    const int size_im2col  = ksize * ksize  * in_channels * width * height;  // size for im2col matrix


    // Buffer for all convolution weights
    anydsl::Array<float> weights(sizeof(float) * (memsize_weights));

    // Buffer for all convolution biases, one for each out_channel
    float* biases = (float*) malloc(sizeof(float) * (memsize_biases));

    // Buffer for in matrix
    anydsl::Array<float> in_mat(sizeof(float) * width * height * in_channels);

    // Buffer for out matrix
    anydsl::Array<float> out_mat(sizeof(float) * (width * height * out_channels + size_im2col));

    // Buffer for ref matrix
    anydsl::Array<float> ref_mat(sizeof(float) * width * height * out_channels);

    read_in_weigths_chw(weights.data(), 0, "../bench/upconv1.txt", 64, 32, 5);
    read_in_biases(biases, 0, "../bench/uc1bias.txt", 32);
    read_in_matrix_chw(in_mat.data(), "../bench/conv4_in.txt", in_channels, height, width);
    read_in_matrix_chw(ref_mat.data(), "../bench/conv4_out.txt", out_channels, height, width);

    auto ticks = std::chrono::high_resolution_clock::now();
    conv_bench(&in_mat, &weights, biases, &out_mat);
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ticks).count();
    std::cout << "Time:\t" << elapsed_ms << " ms" << std::endl;

    for (size_t chn = 0; chn < out_channels; chn++) {
        for (size_t row = 0; row < height; row++) {
            for (size_t col = 0; col < width; col++) {
                if (abs(out_mat.data()[size_im2col + chn * width * height + row * width + col] - ref_mat.data()[chn * width * height + row * width + col]) > 1.0e-4) {
                    std::cout << "Diff at:\t" << chn << "\t" << width << "\t" << height << "\t(chn, x, y)" << std::endl;
                    std::cout << "Was:\t\t" << out_mat.data()[size_im2col + chn * width * height + row * width + col] << "\n"
                        << "Should be:\t" << ref_mat.data()[chn * width * height + row * width + col] << std::endl;
                    goto outer_break;
                }
            }
        }
    }
    std::cout << "Correct Calculation!" << std::endl;
outer_break:

    in_mat.release();
    out_mat.release();
    ref_mat.release();
    weights.release();
    free(biases);
}

/*
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

void denoise(std::string path1, std::string path2, std::string path3) {
    const int memsize1  = 3 * 3 *  9 * 12;
    const int memsize2  = 3 * 3 * 12 * 12;
    const int memsize3  = 3 * 3 * 12 * 16;
    const int memsize4  = 3 * 3 * 16 * 32;
    const int memsize5  = 3 * 3 * 32 * 64;
    const int memsize6  = 3 * 3 * 64 * 70;
    const int memsize7  = 3 * 3 * 70 * 70;
    const int memsize8  = 3 * 3 * 102* 92;
    const int memsize9  = 3 * 3 * 92 * 92;
    const int memsize10 = 3 * 3 * 108* 70;
    const int memsize11 = 3 * 3 * 70 * 70;
    const int memsize12 = 3 * 3 * 82 * 64;
    const int memsize13 = 3 * 3 * 64 * 64;
    const int memsize14 = 3 * 3 * 73 * 32;
    const int memsize15 = 3 * 3 * 32 * 16;
    const int memsize16 = 3 * 3 * 16 * 3;

    const int memsize_biases = 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70 + 70 + 64 + 64 + 32 + 16 + 3;
    const int memsize_weights = (memsize1 + memsize2  + memsize3  + memsize4  + memsize5  + memsize6  + memsize7  + memsize8 +
                                memsize9  + memsize10 + memsize11 + memsize12 + memsize13 + memsize14 + memsize15 + memsize16);

    // Buffer for all convolution weights
    anydsl::Array<float> weights(sizeof(float) * memsize_weights);

    float* biases = (float*) malloc(sizeof(float) * (memsize_biases));

    int offset = 0;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv1.txt",  9, 12, 3);
    offset += memsize1;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv2.txt", 12, 12, 3);
    offset += memsize2;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv3.txt", 12, 16, 3);
    offset += memsize3;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv4.txt", 16, 32, 3);
    offset += memsize4;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv5.txt", 32, 64, 3);
    offset += memsize5;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv6.txt", 64, 70, 3);
    offset += memsize6;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv7.txt", 70, 70, 3);
    offset += memsize7;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv8.txt", 102, 92, 3);
    offset += memsize8;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv9.txt", 92, 92, 3);
    offset += memsize9;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv10.txt", 108, 70, 3);
    offset += memsize10;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv11.txt", 70, 70, 3);
    offset += memsize11;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv12.txt", 82, 64, 3);
    offset += memsize12;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv13.txt", 64, 64, 3);
    offset += memsize13;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv14.txt", 73, 32, 3);
    offset += memsize14;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv15.txt", 32, 16, 3);
    offset += memsize15;
    read_in_weigths_chw(weights.data(), offset, "/home/woshi/Documents/mcdenoise/network/conv16.txt", 16, 3, 3);

    read_in_biases(biases, 0, "/home/woshi/Documents/mcdenoise/network/bias1.txt",  12);
    read_in_biases(biases, 12, "/home/woshi/Documents/mcdenoise/network/bias2.txt",  12);
    read_in_biases(biases, 12 + 12, "/home/woshi/Documents/mcdenoise/network/bias3.txt",  16);
    read_in_biases(biases, 12 + 12 + 16, "/home/woshi/Documents/mcdenoise/network/bias4.txt", 32);
    read_in_biases(biases, 12 + 12 + 16 + 32, "/home/woshi/Documents/mcdenoise/network/bias5.txt",  64);
    read_in_biases(biases, 12 + 12 + 16 + 32 + 64, "/home/woshi/Documents/mcdenoise/network/bias6.txt",  70);
    read_in_biases(biases, 12 + 12 + 16 + 32 + 64 + 70, "/home/woshi/Documents/mcdenoise/network/bias7.txt",  70);
    read_in_biases(biases, 12 + 12 + 16 + 32 + 64 + 70 + 70, "/home/woshi/Documents/mcdenoise/network/bias8.txt",  92);
    read_in_biases(biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92, "/home/woshi/Documents/mcdenoise/network/bias9.txt",  92);
    read_in_biases(biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92, "/home/woshi/Documents/mcdenoise/network/bias10.txt",  70);
    read_in_biases(biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70, "/home/woshi/Documents/mcdenoise/network/bias11.txt",  70);
    read_in_biases(biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70 + 70, "/home/woshi/Documents/mcdenoise/network/bias12.txt",  64);
    read_in_biases(biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70 + 70 + 64, "/home/woshi/Documents/mcdenoise/network/bias13.txt",  64);
    read_in_biases(biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70 + 70 + 64 + 64, "/home/woshi/Documents/mcdenoise/network/bias14.txt",  32);
    read_in_biases(biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70 + 70 + 64 + 64 + 32, "/home/woshi/Documents/mcdenoise/network/bias15.txt",  16);
    read_in_biases(biases, 12 + 12 + 16 + 32 + 64 + 70 + 70 + 92 + 92 + 70 + 70 + 64 + 64 + 32 + 16, "/home/woshi/Documents/mcdenoise/network/bias16.txt",  3);

    std::cout << "Memory in cpp:\t\t" << 4 * (memsize_biases + memsize_weights) << std::endl;

    ImageRgba32 noisy;
    load_png(FilePath(path1), noisy);
    int width  = noisy.width;
    int height = noisy.height;

    ImageRgba32 albedo;
    load_png(FilePath(path2), albedo);

    ImageRgba32 normal;
    load_png(FilePath(path3), normal);

    anydsl::Array<uint8_t> result(sizeof(uint8_t) * width * height * 3);

    std::cout << "Memory in cpp:\t\t" << 4 * (memsize_biases + memsize_weights) + sizeof(uint8_t) * (2 * width) * (2 * height) * 3 + width * height * 3 << std::endl;

    denoise_im2col(&noisy.pixels, &albedo.pixels, &normal.pixels, width, height, &result, &weights, biases);

    save_png_pointer(FilePath("out_im2col.png"), result.data(), width, height);

    // Free all allocated memory
    weights.release();
    noisy.pixels.release();
    albedo.pixels.release();
    normal.pixels.release();
    result.release();
    free(biases);
}
*/
