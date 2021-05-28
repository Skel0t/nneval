#ifndef IMAGE_H
#define IMAGE_H

#include <memory>

#include <anydsl_runtime.hpp>

#include "file_path.h"

struct ImageRgba32 {
    anydsl::Array<uint8_t> pixels;
    size_t width, height;
};

bool load_png(const FilePath&, ImageRgba32&);
bool save_png(const FilePath&, const ImageRgba32&);
bool save_png_grayscale(const FilePath& path, const uint8_t* pixels, const int width, const int height);
bool save_png_pointer(const FilePath& path, const uint8_t* pixels, const int width, const int height);

#endif // IMAGE_H
