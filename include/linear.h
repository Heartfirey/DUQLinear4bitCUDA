#pragma once

#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
// quarot gemm
#include <gemm.h>

template <typename T>
class Linear
{
public:
    uint32_t input_dim;
    uint32_t output_dim;
    bool use_bias;

    T *weight;
    T *bias;

    Linear(uint32_t input_dim, uint32_t output_dim, bool use_bias)
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

    ~Linear()
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
