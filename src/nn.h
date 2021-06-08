#ifndef NN_H
#define NN_H

float* read_in_weigths(std::string path, int in_channels, int out_channels, int ksize);
anydsl::Array<anydsl::Array<float>*>* read_in_weigths2(std::string path, int in_channels, int out_channels, int ksize);
float* read_in_biases(std::string path, int out_channels);

#endif /* NN_H */