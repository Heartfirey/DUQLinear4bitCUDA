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
