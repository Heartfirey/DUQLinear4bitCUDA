#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/all.h>
#include <ATen/ATen.h>
#include <cmath>
#include <common.h>
#include <quant.h>

class PackedQuantizedTensor
{
public:
    torch::Tensor quantized_x;
    torch::Tensor scale;

    PackedQuantizedTensor(torch::Tensor quantized_x, torch::Tensor scale): quantized_x(quantized_x), scale(scale) {}

    torch::IntArrayRef sizes() const {
        return quantized_x.sizes();
    }
    torch::Device device() const {
        return quantized_x.device();
    }
};

// Packed Quantized Tensor with two scales
class PackedDoubleQuantizedTensor
{
public:
    torch::Tensor quantized_x;
    torch::Tensor scale_1;
    torch::Tensor scale_2;

    PackedDoubleQuantizedTensor(torch::Tensor quantized_x, torch::Tensor scale_1, torch::Tensor scale_2): quantized_x(quantized_x), scale_1(scale_1), scale_2(scale_2) {}

    torch::IntArrayRef sizes() const {
        return quantized_x.sizes();
    }

    torch::Device device() const {
        return quantized_x.device();
    }
};

torch::Tensor sym_quant(const torch::Tensor &x, const torch::Tensor &scale);

torch::Tensor sym_dual_quant(const torch::Tensor &x, const torch::Tensor &scale_1, const torch::Tensor &scale_2);

torch::Tensor sym_dequant(const torch::Tensor &q,
                          const torch::Tensor &scale_row,
                          const torch::Tensor &scale_col,
                          const int bits);

torch::Tensor sym_dual_dequant(const torch::Tensor &q,
                                 const torch::Tensor &scale_row,
                                 const torch::Tensor &scale_col_1,
                                 const torch::Tensor &scale_col_2,
                                 const int bits);

class Quantizer4bit : public torch::nn::Module
{
public:
    float input_clip_ratio;
    Quantizer4bit(float input_clip_ratio) : input_clip_ratio(input_clip_ratio) {}
    PackedQuantizedTensor forward(torch::Tensor x);
};

class DeQuantizer4bit : torch::nn::Module
{
public:
    uint32_t bits = 32;
    DeQuantizer4bit() {}
    DeQuantizer4bit(uint32_t bits) : bits(bits) {}

    // torch::Tensor forward(PackedQuantizedTensor packed_tensor, torch::Tensor weight_scale);
    torch::Tensor forward(torch::Tensor &x, torch::Tensor &scale_x, torch::Tensor &weight_scale);
};

class DoubleQuantizer4bit : public torch::nn::Module
{
    //TODO: Finsh this class
};

class DoubleDeQuantizer4bit : public torch::nn::Module
{
    //TODO: Finish this class
};

torch::Tensor pack_i4(torch::Tensor q);
