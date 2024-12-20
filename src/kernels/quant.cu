#include <quant.h>


template<typename T>
__device__ __half int_to_half(T value)
{
    return __int2half_rn(static_cast<int>(value));
}


__global__
void sym_quantize_f16_i4_kernel(
        const half *__restrict__ x,
        const half *__restrict__ scale,
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
    memset(&storage, 0, sizeof(storage));
    uint32_t id = colDst * kElementsPerVector + row * colsSrc;
#pragma unroll
    for (int i = 0; i < kElementsPerVector; ++i)
    {
        bool safe = (colDst * kElementsPerVector + i) < colsSrc;
        if (safe)
        {
            half data = __hdiv(x[id + i], scale[row]);

            int qval = clamp(__half2int_rn(data), qmin_sym, qmax_sym);
            Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.set(
                    qval);
        }
    }

    q[colDst + row * colsDst] = storage;
}


void sym_quant_host(
        const half *x,
        const half *scale,
        uint32_t rows,
        uint32_t colsSrc,
        uint32_t colsDst,
        Int4Storage *q
)
{

    dim3 block{std::min<uint32_t>(colsDst, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(colsDst, block.x), cdiv(rows, block.y)};
    sym_quantize_f16_i4_kernel<<<grid, block>>>(x, scale, rows, colsSrc, colsDst, q);
}


__global__ void sym_dequantize_i32_f16_kernel(
        const int32_t *__restrict__ q,
        const half *__restrict__ scale_row,
        const half *__restrict__ scale_col,
        uint32_t rows, uint32_t cols,
        half *__restrict__ x)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

    if (col >= cols || row >= rows)
    {
        return;
    }

    half xElement = int_to_half(q[col + row * cols]);
    x[col + row * cols] = scale_row[row] * scale_col[col] * xElement;
}

void sym_dequant_host(const int32_t *q,
                                 const half *scale_row,
                                 const half *scale_col,
                                 uint32_t rows,
                                 uint32_t cols,
                                 half *x
)
{
    dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
    sym_dequantize_i32_f16_kernel<<<grid, block>>>(
            q,
            scale_row, scale_col,
            rows, cols, x);
}

__global__
void asym_quantize_f16_i4_kernel(
    const half *__restrict__ x,
    const half *__restrict__ scale,
    const half *__restrict__ zeros,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    UInt4Storage *__restrict__ q
)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t colDst = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows || colDst * kElementsPerVectorUInt4 >= colsSrc)
    {
        return;
    }
    UInt4Storage storage;
    memset(&storage, 0, sizeof(storage));
    uint32_t id = colDst * kElementsPerVectorUInt4 + row * colsSrc;
    #pragma unroll
    for (int i = 0; i < kElementsPerVectorUInt4; ++i)
    {
        bool safe = (colDst * kElementsPerVectorUInt4 + i) < colsSrc;
        if (safe)
        {
            half data = __hdiv(x[id + i], scale[row]);
            int qval = clamp(__half2int_rn(data) + __half2int_rn(zeros[row]), qmin_asym, qmax_asym);
            UInt4Subbyte(reinterpret_cast<cutlass::uint4b_t *>(&storage), i).set(qval);
        }
    }
    
    q[colDst + row * colsDst] = storage;
}

void asym_quant_host(
    const half *x,
    const half *scale,
    const half *zeros,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    UInt4Storage *q
)
{
    dim3 block{std::min<uint32_t>(colsDst, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(colsDst, block.x), cdiv(rows, block.y)};
    asym_quantize_f16_i4_kernel<<<grid, block>>>(x, scale, zeros, rows, colsSrc, colsDst, q);
}

__global__
void asym_dequantize_i32_f16_kernel(
    const int32_t *__restrict__ q,
    const half *__restrict__ scale_row,
    const half *__restrict__ zeros_row,
    const half *__restrict__ scale_col,
    const half *__restrict__ zeros_col,
    uint32_t rows,
    uint32_t cols,
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
    x[col + row * cols] = (scale_row[row] * scale_col[col]) * (xElement - zeros_row[row] - zeros_col[col]);
}

void asym_dequant_host(
    const int32_t *q,
    const half *scale_row,
    const half *zeros_row,
    const half *scale_col,
    const half *zeros_col,
    uint32_t rows,
    uint32_t cols,
    half *x
)
{
    dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
    asym_dequantize_i32_f16_kernel<<<grid, block>>>(q, 
                                                    scale_row, zeros_row, 
                                                    scale_col, zeros_col, 
                                                    rows, cols, x);
}

__global__
void asym_batch_dequantize_i32_f16_kernel(
    const int32_t *__restrict__ q,
    const half *__restrict__ scale_row,
    const half *__restrict__ zeros_row,
    const half *__restrict__ scale_col,
    const half *__restrict__ zeros_col,
    uint32_t batches,
    uint32_t rows,
    uint32_t cols,
    half *x
)
{
    uint32_t batch = threadIdx.z + blockIdx.z * blockDim.z;
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;
    if (batch >= batches || col >= cols || row >= rows)
    {
        return;
    }

    uint32_t q_index = col + row * cols + batch * cols * rows;
    uint32_t scale_row_index = row + batch * rows;
    uint32_t scale_col_index = col + batch * cols;
    uint32_t x_index = col + row * cols + batch * cols * rows;

    half xElement = int_to_half(q[q_index]);
    x[x_index] = (scale_row[scale_row_index] * scale_col[scale_col_index]) *
                 (xElement - zeros_row[scale_row_index] - zeros_col[scale_col_index]);
}

void asym_batch_dequant_host(
    const int32_t *q,
    const half *scale_row,
    const half *zeros_row,
    const half *scale_col,
    const half *zeros_col,
    uint32_t batches,
    uint32_t rows,
    uint32_t cols,
    half *x
)
{
    dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16), std::min<uint32_t>(batches, 4)};
    dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y), cdiv(batches, block.z)};
    asym_batch_dequantize_i32_f16_kernel<<<grid, block>>>(q, 
                                                          scale_row, zeros_row, 
                                                          scale_col, zeros_col, 
                                                          batches, rows, cols, x);
}

constexpr int32_t qmin_asym_8bit = 0;
constexpr int32_t qmax_asym_8bit = 126;

__global__
void asym_quantize_f16_i8_kernel(
    const float *__restrict__ x,
    const float *__restrict__ scale,
    const float *__restrict__ zeros,
    uint32_t rows,
    uint32_t cols,
    int8_t *__restrict__ q
)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows || col >= cols)
    {
        return;
    }

    uint32_t id = col + row * cols;
    float data = (x[id] / scale[row]);
    int qval = clamp(__half2int_rn(data) + __half2int_rn(zeros[row]), qmin_asym_8bit, qmax_asym_8bit);
    q[id] = qval;
}

void asym_quant_host_8bit(
    const float *x,
    const float *scale,
    const float *zeros,
    uint32_t rows,
    uint32_t cols,
    int8_t *q 
)
{
    dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
    asym_quantize_f16_i8_kernel<<<grid, block>>>(x, scale, zeros, rows, cols, q);
}

__global__
void asym_dequantize_i32_f16_hp_kernel(
    const int32_t *__restrict__ q,
    const float *__restrict__ scale_row,
    const float *__restrict__ zeros_row,
    const float *__restrict__ scale_col,
    const float *__restrict__ zeros_col,
    uint32_t rows,
    uint32_t cols,
    float *__restrict__ x
)
{
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;
    if (col >= cols || row >= rows)
    {
        return;
    }

    float xElementF = (float)(q[col + row * cols]);
    float scales = (scale_row[row]) * (scale_col[col]);
    float zeros = (zeros_row[row]) + (zeros_col[col]);
    x[col + row * cols] = (scales * (xElementF - zeros));
}

void asym_dequant_host_hprec(
    const int32_t *q,
    const float *scale_row,
    const float *zeros_row,
    const float *scale_col,
    const float *zeros_col,
    uint32_t rows,
    uint32_t cols,
    float *x
)
{
    dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16)};
    dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
    asym_dequantize_i32_f16_hp_kernel<<<grid, block>>>(q, 
                                                        scale_row, zeros_row, 
                                                        scale_col, zeros_col, 
                                                        rows, cols, x);
}
