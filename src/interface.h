#ifndef INTERFACE_H
#define INTERFACE_H
#include <anydsl_runtime.hpp>

#ifdef __cplusplus
extern "C" {
#endif

void sparse_mult_test(anydsl::Array<float>*, anydsl::Array<int>*, anydsl::Array<float>*);
void create_conv_test(anydsl::Array<float>*, anydsl::Array<float>*);
uint8_t* image_kernel_test(anydsl::Array<float>*, anydsl::Array<float>*, anydsl::Array<float>*, anydsl::Array<uint8_t>*);
void sres(anydsl::Array<uint8_t>* in, int32_t width, int32_t height, anydsl::Array<uint8_t>* out, anydsl::Array<anydsl::Array<float>*>* kernels1, anydsl::Array<anydsl::Array<float>*>* kernels2, anydsl::Array<anydsl::Array<float>*>* kernels3,
          anydsl::Array<anydsl::Array<float>*>* upkernels1, anydsl::Array<anydsl::Array<float>*>* kernels4, anydsl::Array<anydsl::Array<float>*>* kernels5, anydsl::Array<anydsl::Array<float>*>* kernels6,
          float* biases1 , float* biases2 , float* biases3 , float* upbiases1 , float* biases4 , float* biases5 , float* biases6);
void offset_matrix_test(anydsl::Array<float>* Buffer, float* biases);

#ifdef __cplusplus
}
#endif

#endif /* INTERFACE_H */
