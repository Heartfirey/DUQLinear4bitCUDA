#pragma once

#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
// cuda random
#include <curand.h>
#include <cuda_fp16.h>
// quarot gemm
#include <common.h>
#include <gemm.h>
#include <quant.h>
#include <quant_tensor.h>
#include <torch/all.h>
#include <logger.h>

class Linear : public torch::nn::Module
{
public:
    torch::Tensor weight;
    torch::Tensor bias;

    Linear(uint32_t input_dim, uint32_t output_dim, bool bias, torch::Dtype dtype);
    // ~Linear();

    torch::Tensor forward(torch::Tensor input);
};


class Linear4bit : public torch::nn::Module
{
public:
    torch::Tensor weight;
    torch::Tensor weight_scale;
    torch::Tensor bias;

    Linear4bit(uint32_t input_dim, uint32_t output_dim, bool bias, torch::Dtype dtype=torch::kFloat16);
    // ~Linear4bit();

    torch::Tensor forward(PackedQuantizedTensor input);
    static Linear4bit from_float(Linear &linear, torch::Tensor weight_scales);
    static Linear4bit from_float(torch::nn::Linear &linear, torch::Tensor weight_scales);
};

// Linea4bit module with dual uniform quantization
class Linear4bitDUQ : public torch::nn::Module
{
public:
    torch::Tensor weight;
    torch::Tensor weight_scale_1;
    torch::Tensor weight_scale_2;
    torch::Tensor bias;

    Linear4bitDUQ(uint32_t input_dim, uint32_t output_dim, bool bias, torch::Dtype dtype=torch::kFloat16);

    torch::Tensor forward(PackedQuantizedTensor input);
    static Linear4bitDUQ from_float(Linear &linear, torch::Tensor weight_scales_1, torch::Tensor weight_scales_2);
    static Linear4bitDUQ from_float(torch::nn::Linear &linear, torch::Tensor weight_scales_1, torch::Tensor weight_scales_2);
};

template <typename T>
class LinearPlain
{
public:
    uint32_t input_dim;
    uint32_t output_dim;
    bool use_bias;

    T *weight;
    float *bias;

    LinearPlain(uint32_t input_dim, uint32_t output_dim, bool use_bias)
        : input_dim(input_dim), output_dim(output_dim), use_bias(use_bias)
    {
        uint32_t weight_size = input_dim * output_dim * sizeof(T);

        cudaMalloc(&weight, weight_size);
        if (use_bias)
        {
            cudaMalloc(&bias, output_dim * sizeof(T));
        }

        // initialize_weights();
        if (use_bias)
        {
            initialize_bias();
        }
    }

    ~LinearPlain()
    {
        cudaFree(weight);
        if (use_bias)
        {
            cudaFree(bias);
        }
    }

    void initialize_weights()
    {
        float stddev = sqrt(2.0 / input_dim);
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 123ULL);
        curandGenerateNormal(gen, weight, input_dim * output_dim, 0.0f, stddev);
    }

    void initialize_bias()
    {
        cudaMemset(bias, 0, output_dim * sizeof(T));
    }

    void forward(T *input, T *output, uint32_t batch_size)
    {
        // B->M, IN_DIM->K, OUT_DIM->N
        // input shpae: (B, IN_DIM) 
        // weight shape: (OUT_DIM, IN_DIM) 
        // output shape: (B, OUT_DIM)
        uint32_t M = batch_size;
        uint32_t N = output_dim;
        uint32_t K = input_dim;
        matmul_host(input, weight, M, N, K, output);
        if (use_bias)
        {
            add_bias<<<(output_dim + 255) / 256, 256>>>(output, bias, batch_size, output_dim);
        }
    }
};

// TODO: implement Linear4bitPlain
// class Linear4bitPlain
// {

// };
