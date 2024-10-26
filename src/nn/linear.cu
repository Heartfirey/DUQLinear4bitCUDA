#pragma once

#include <iostream>
#include <curand.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
// quarot gemm
#include <gemm.h>
#include <int4.h>

class Linear4bit
{
public:
    uint32_t input_dim;
    uint32_t output_dim;
    bool use_bias;
    Int4Storage *weight;
    Int4Storage *bias;

    
}

