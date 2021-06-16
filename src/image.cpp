#include <png.h>
#include <fstream>
#include <cmath>

#include "image.h"

static void gamma_correct(ImageRgba32& img) {
    for (size_t y = 0; y < img.height; ++y) {
        for (size_t x = 0; x < img.width; ++x) {
            auto* pix = &img.pixels[3 * (y * img.width + x)];
            for (int i = 0; i < 3; ++i)
                pix[i] = std::pow(pix[i] * (1.0f / 255.0f), 2.2f) * 255.0f;
        }
    }
}

static void read_from_stream(png_structp png_ptr, png_bytep data, png_size_t length) {
    png_voidp a = png_get_io_ptr(png_ptr);
    ((std::istream*)a)->read((char*)data, length);
}

bool load_png(const FilePath& path, ImageRgba32& img) {
    std::ifstream file(path, std::ifstream::binary);
    if (!file)
        return false;

    // Read signature
    char sig[8];
    file.read(sig, 8);
    if (!png_check_sig((unsigned char*)sig, 8))
        return false;

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr)
        return false;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        return false;
    }

    png_set_sig_bytes(png_ptr, 8);
    png_set_read_fn(png_ptr, (png_voidp)&file, read_from_stream);
    png_read_info(png_ptr, info_ptr);

    img.width    = png_get_image_width(png_ptr, info_ptr);
    img.height   = png_get_image_height(png_ptr, info_ptr);

    png_uint_32 color_type = png_get_color_type(png_ptr, info_ptr);
    png_uint_32 bit_depth  = png_get_bit_depth(png_ptr, info_ptr);

    // Expand paletted and grayscale images to RGB
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png_ptr);
    } else if (color_type == PNG_COLOR_TYPE_GRAY ||
               color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png_ptr);
    }

    // Transform to 8 bit per channel
    if (bit_depth == 16)
        png_set_strip_16(png_ptr);

    // Get alpha channel when there is one
    // if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
    //     png_set_tRNS_to_alpha(png_ptr);

    // // Otherwise add an opaque alpha channel
    // else
    //     png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER);

    img.pixels = anydsl::Array<uint8_t>(sizeof(uint8_t) * img.width * img.height * 3);
    std::unique_ptr<png_byte[]> row_bytes(new png_byte[img.width * 3]);
    for (size_t y = 0; y < img.height; y++) {
        png_read_row(png_ptr, row_bytes.get(), nullptr);
        auto img_row = img.pixels.data() + 3 * img.width * y; // (img.height - 1 - y);
        for (size_t x = 0; x < img.width; x++) {
            for (size_t c = 0; c < 3; ++c)
                img_row[x * 3 + c] = row_bytes[x * 3 + c];
        }
    }

    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    // gamma_correct(img);
    return true;
}

static void png_write_to_stream(png_structp png_ptr, png_bytep data, png_size_t length) {
    png_voidp a = png_get_io_ptr(png_ptr);
    ((std::ostream*)a)->write((const char*)data, length);
}

static void png_flush_stream(png_structp) {
    // Nothing to do
}

bool save_png(const FilePath& path, const ImageRgba32& img) {
    std::ofstream file(path, std::ofstream::binary);
    if (!file)
        return false;

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr)
        return false;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        return false;
    }

    std::unique_ptr<uint8_t[]> row_bytes(new uint8_t[img.width * 4]);
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }

    png_set_write_fn(png_ptr, &file, png_write_to_stream, png_flush_stream);

    png_set_IHDR(png_ptr, info_ptr, img.width, img.height,
                 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    for (size_t y = 0; y < img.height; y++) {
        auto img_row = ((img.pixels.data()) + 4 * img.width * (img.height - 1 - y));
        for (size_t x = 0; x < img.width; x++) {
            for (size_t c = 0; c < 4; ++c)
                row_bytes[x * 4 + c] = img_row[x * 4 + c];
        }
        png_write_row(png_ptr, row_bytes.get());
    }

    png_write_end(png_ptr, info_ptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    return true;
}

bool save_png_pointer(const FilePath& path, const uint8_t* pixels, const int width, const int height) {
    std::ofstream file(path, std::ofstream::binary);
    if (!file)
        return false;

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr)
        return false;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        return false;
    }

    std::unique_ptr<uint8_t[]> row_bytes(new uint8_t[width * 4]);
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }

    png_set_write_fn(png_ptr, &file, png_write_to_stream, png_flush_stream);

    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    for (size_t y = 0; y < height; y++) {
        auto img_row = pixels + 3 * width * y;
        for (size_t x = 0; x < width; x++) {
            row_bytes[x * 4 + 0] = img_row[3*x];
            row_bytes[x * 4 + 1] = img_row[3*x+1];
            row_bytes[x * 4 + 2] = img_row[3*x+2];
            row_bytes[x * 4 + 3] = 255;
        }
        png_write_row(png_ptr, row_bytes.get());
    }

    png_write_end(png_ptr, info_ptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    return true;
}

bool save_png_grayscale(const FilePath& path, const uint8_t* pixels, const int width, const int height) {
    std::ofstream file(path, std::ofstream::binary);
    if (!file)
        return false;

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr)
        return false;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        return false;
    }

    std::unique_ptr<uint8_t[]> row_bytes(new uint8_t[width * 4]);
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }

    png_set_write_fn(png_ptr, &file, png_write_to_stream, png_flush_stream);

    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    for (size_t y = 0; y < height; y++) {
        auto img_row = pixels + width * y;
        for (size_t x = 0; x < width; x++) {
            row_bytes[x * 4 + 0] = img_row[x];
            row_bytes[x * 4 + 1] = img_row[x];
            row_bytes[x * 4 + 2] = img_row[x];
            row_bytes[x * 4 + 3] = 255;
        }
        png_write_row(png_ptr, row_bytes.get());
    }

    png_write_end(png_ptr, info_ptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    return true;
}
