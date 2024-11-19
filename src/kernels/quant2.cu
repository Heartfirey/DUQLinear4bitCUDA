#include <quant.h>

template<typename T>
__device__ __half int_to_half(T value)
{
    return __int2half_rn(static_cast<int>(value));
}

__global__
void sym_quantize_f16_i4_scale2_kernel(
    const half *__restrict__ x,
    const half *__restrict__ scale_1,
    const half *__restrict__ scale_2,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    Int4Storage *__restrict__ q
)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t colDst = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows || colDst * kElementsPerVector >= colsSrc)
    {
        return;
    }
    Int4Storage storage;
    memset(&storage, 0, sizeof(Int4Storage));
    uint32_t id = colDst * kElementsPerVector + row * colsSrc;
    #pragma unroll
    for (int i = 0; i < kElementsPerVector; ++i) 
    {
        bool safe = (colDst * kElementsPerVector + i) < colsSrc;
        if (safe)
        {
            // Decide the current col quantize scale
            half scale_val = (colDst * kElementsPerVector + i < QUANT_COL_K) ? (scale_1[row]) : (scale_2[row]);
            half data = __hdiv(x[id + i], scale_val);

            int qval = clamp(__half2int_rn(data), qmin, qmax);
            Int4Subbyte(reinterpret_cast<cutlass::int4b_t *>(&storage), i).set(qval);
        }

    }
    q[colDst + row * colsDst] = storage;
}

void sym_double_quant_host(
    const half *x,
    const half *scale_1,
    const half *scale_2,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    Int4Storage *q
)
{
    dim3 block{std::min<uint32_t>(colsDst, 32), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(colsDst, block.x), cdiv(rows, block.y)};
    sym_quantize_f16_i4_scale2_kernel<<<grid, block>>>(x, scale_1, scale_2, rows, colsSrc, colsDst, q);
}

__global__
void sym_dequantize_i32_f16_kernel(
    const int32_t *__restrict__ q,
    const half *__restrict__ scale_row,
    const half *__restrict__ scale_col_1,
    const half *__restrict__ scale_col_2,
    uint32_t rows, uint32_t cols,
    half *__restrict__ x
)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

    if (col >= cols || row >= rows)
    {
        return;
    }

    half xElement = int_to_half(q[col + row * cols]);
    half scale_col_used = col < QUANT_COL_K ? scale_col_1[col] : scale_col_2[col];
    x[col + row * cols] = scale_row[row] * scale_col_used * xElement;
}

void sym_double_dequant_host(
    const int32_t *q,
    const half *scale_row,
    const half *scale_col_1,
    const half *scale_col_2,
    uint32_t rows,
    uint32_t cols,
    half *x
)
{
    dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
    sym_dequantize_i32_f16_kernel<<<grid, block>>>(q, scale_row, scale_col_1, scale_col_2, rows, cols, x);
}



