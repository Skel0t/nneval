#ifndef INTERFACE_H
#define INTERFACE_H
#include <anydsl_runtime.hpp>

#ifdef __cplusplus
extern "C" {
#endif

void sparse_mult_test(anydsl::Array<float>*, anydsl::Array<int>*, anydsl::Array<float>*);
void create_conv_test(anydsl::Array<float>*, anydsl::Array<float>*);
void image_kernel_test(anydsl::Array<float>* kernel, anydsl::Array<uint8_t>* in, anydsl::Array<uint8_t>* out);
void sres(anydsl::Array<uint8_t>* in, int32_t width, int32_t height, anydsl::Array<uint8_t>* out, anydsl::Array<float>* kernels, float* biases);
void offset_matrix_test(anydsl::Array<float>* Buffer, float* biases);

#ifdef __cplusplus
}
#endif

#endif /* INTERFACE_H */
