#ifndef INTERFACE_H
#define INTERFACE_H
#include <anydsl_runtime.hpp>

#ifdef __cplusplus
extern "C" {
#endif

void image_kernel_test(anydsl::Array<float>* kernel, anydsl::Array<uint8_t>* in, anydsl::Array<uint8_t>* out);
void sres_im2col(anydsl::Array<uint8_t>* img, int32_t width, int32_t height, anydsl::Array<uint8_t>* out, anydsl::Array<float>* kernels, float* biases);
void denoise_im2col(anydsl::Array<uint8_t>* img, anydsl::Array<uint8_t>* alb, anydsl::Array<uint8_t>* nrm, int32_t width, int32_t height, anydsl::Array<uint8_t>* out, anydsl::Array<float>* kernels, float* biases);
void sres_conv_mat(anydsl::Array<uint8_t>* in, int32_t width, int32_t height, anydsl::Array<uint8_t>* out, anydsl::Array<float>* kernels, float* biases);
void im2col_test(anydsl::Array<uint8_t>* in, int32_t width, int32_t height, anydsl::Array<float>* kernels, anydsl::Array<uint8_t>* out);
void conv_bench(anydsl::Array<float>* in_mat, anydsl::Array<float>* flattened_kernels, float* biases, anydsl::Array<float>* out);

#ifdef __cplusplus
}
#endif

#endif /* INTERFACE_H */
