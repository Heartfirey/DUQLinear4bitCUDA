#pragma once

#include <common.h>
#include <cutools.cuh>

template <typename T>
void matmul_host(
    const T *A,
    const T *B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    T *C
)
{
    using Gemm = cutlass::gemm::device::Gemm<
        T,                              // ElementA
        cutlass::layout::RowMajor,      // LayoutA
        T,                              // ElementB
        cutlass::layout::ColumnMajor,   // LayoutB
        T,                              // ElementOutput
        cutlass::layout::RowMajor,      // LayoutOutput
        T,                              // ElementAccumulator
        cutlass::arch::OpClassSimt,     // tag indicating Tensor Cores
        cutlass::arch::Sm80             // GPU Compute Arch(Lock to 3090-Ampere)
    >;

    Gemm gemmOp;

    using GemmCoord = cutlass::gemm::GemmCoord;

    typename Gemm::Arguments arguments {
        {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N), static_cast<GemmCoord::Index>(K)},
        {(T *)A,    K},
        {(T *)B,    K},
        {C,         N},
        {C,         N},
        {1,         0}
    };

    CHECK_CUTLASS(gemmOp(arguments));
}

void matmul_host_4bit(
        const Int4Storage *A,
        const Int4Storage *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
);

template <typename T>
__global__ void add_bias(
    T *C,
    T *bias,
    uint32_t batch_size,
    uint32_t feat_dim
)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < feat_dim)
    {
        for (int i = 0; i < batch_size; i++)
        {
            C[i * feat_dim + idx] += bias[idx];
        }
    }
}
