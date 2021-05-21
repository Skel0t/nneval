#include <iostream>

#include "nn.h"
#include "image.h"

void sparseMult();
void imageTest();
void create_conv();

int main() {
    create_conv();

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
    ImageRgba32 img;
    FilePath path("/home/woshi/Documents/nneval/src/mitchell.png");
    load_png(path, img);
    save_png(FilePath("out.png"), img);
}