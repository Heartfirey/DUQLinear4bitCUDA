#include <cutlass/gemm/device/gemm.h>
#include <cutools.cuh>
#include <gemm.h>


void matmul_host_4bit(
        const Int4Storage *A,
        const Int4Storage *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
)
{
    using Gemm = cutlass::gemm::device::Gemm<
            cutlass::int4b_t,                // ElementA
            cutlass::layout::RowMajor,       // LayoutA
            cutlass::int4b_t,                // ElementB
            cutlass::layout::ColumnMajor,    // LayoutB
            int32_t,                         // ElementOutput
            cutlass::layout::RowMajor,       // LayoutOutput
            int32_t,                         // ElementAccumulator
            cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
            cutlass::arch::Sm80  // tag indicating target GPU compute architecture  // TODO: This is just for compiling on my laptop temporarily. Should be higher when doing benchmarking.
    >;

    Gemm gemmOp;

    using GemmCoord = cutlass::gemm::GemmCoord;

    typename Gemm::Arguments arguments{
            {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N), static_cast<GemmCoord::Index>(K)},
            {(cutlass::int4b_t *) A,           K},
            {(cutlass::int4b_t *) B,           K},
            {C,                                N},
            {C,                                N},
            {1,                                0}
    };

    CHECK_CUTLASS(gemmOp(arguments));

}

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

template void matmul_host<int>(
    const int *A,
    const int *B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int *C
);
template void matmul_host<half>(
    const half *A,
    const half *B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    half *C
);
template void matmul_host<float>(
    const float *A,
    const float *B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    float *C
);

template __global__ void add_bias<int>(
    int *C,
    int *bias,
    uint32_t batch_size,
    uint32_t feat_dim
);
template __global__ void add_bias<half>(
    half *C,
    half *bias,
    uint32_t batch_size,
    uint32_t feat_dim
);
template __global__ void add_bias<float>(
    float *C,
    float *bias,
    uint32_t batch_size,
    uint32_t feat_dim
);
