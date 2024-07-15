#pragma once

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#include "math.h"

#include <cstdlib>
#include <iostream>

class RtwImage {
public:
    RtwImage() {}

    RtwImage(const char* image_filename) {
        auto filename = std::string(image_filename);

        if (load(filename)) return;
        if (load("images/" + filename)) return;
        if (load("../images/" + filename)) return;
        if (load("../../images/" + filename)) return;
        if (load("../../../images/" + filename)) return;
        if (load("../../../../images/" + filename)) return;
        if (load("../../../../../images/" + filename)) return;
        if (load("../../../../../../images/" + filename)) return;

        std::cerr << "ERROR: Could not load image file '" << image_filename << "'.\n";
    }

    RtwImage& operator=(const RtwImage& image) {
        bdata = image.bdata;
        image_width = image.image_width;
        image_height = image.image_height;
        bytes_per_scanline = image.bytes_per_scanline;

        return *this;
    }

    void free_image() {
        delete[] bdata;
    }

    int get_total_bytes() const {
        return image_height * bytes_per_scanline;
    }

    bool load(const std::string& filename) {
        auto n = bytes_per_pixel;
        float* fdata = stbi_loadf(filename.c_str(), &image_width, &image_height, &n, bytes_per_pixel);
        if (fdata == nullptr) {
            return false;
        }

        bytes_per_scanline = image_width * bytes_per_pixel;
        int total_bytes = get_total_bytes();
        bdata = new unsigned char[total_bytes];

        auto* bptr = bdata;
        auto* fptr = fdata;
        for (auto i = 0; i < total_bytes; i++, fptr++, bptr++) {
            *bptr = float_to_byte(*fptr);
        }

        STBI_FREE(fdata);

        return true;
    }

    __device__ int width()  const { 
        return image_width; 
    }

    __device__ int height() const {
        return image_height;
    }

    __device__ const unsigned char* pixel_data(int x, int y) const {
        x = clamp(x, 0, image_width);
        y = clamp(y, 0, image_height);

        return bdata + y * bytes_per_scanline + x * bytes_per_pixel;
    }

    int            image_width = 0;
    int            image_height = 0;
    int            bytes_per_scanline = 0;
    unsigned char* bdata = nullptr;

private:
    const int      bytes_per_pixel = 3;

    // TODO: Color class
    static unsigned char float_to_byte(float value) {
        if (value <= 0.0) {
            return 0;
        }
        if (1.0 <= value) {
            return 255;
        }
        return static_cast<unsigned char>(256.0 * value);
    }
};