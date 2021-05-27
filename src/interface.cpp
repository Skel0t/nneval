#include <stdint.h>
#include <iostream>

#include "image.h"

extern "C" {

void nn_load_png(int32_t dev, const char* file, uint8_t** pixels, int32_t* width, int32_t* height) {
    ImageRgba32 img;

    load_png(FilePath(file), img);
    *pixels = (img.pixels.data());
    std::cout << img.pixels.size() << std::endl;
    *width  = img.width;
    *height = img.height;
}

}