#include <torch/extension.h>

// Include all files
#include <gemm.h>
#include <quant.h>
#include <flashinfer.h>


torch::Tensor matmul(const torch::Tensor &A, const torch::Tensor &B)
{
    torch::checkAllContiguous("matmul", {{A, "A",       0},
                                                {B, "B", 1}});
    torch::checkDeviceType("matmul", {A, B}, at::DeviceType::CUDA);

    torch::checkAllSameGPU("matmul", {{A, "A",       0},
                                          {   B, "B", 1}});
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    uint32_t K = A.size(1) * kElementsPerVector;  // 4bit packing is on the columns
    auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

    matmul_host_4bit(A.data_ptr<Int4Storage>(), B.data_ptr<Int4Storage>(), M, N, K, C.data_ptr<int32_t>());

    return C;
}

// ===== Sym Quant/Dequant ======

torch::Tensor sym_quant(const torch::Tensor &x, const torch::Tensor &scale)
{
    torch::checkAllContiguous("sym_quant", {{x,     "x",     0},
                                                      {scale, "scale", 1}});
    torch::checkDeviceType("sym_quant", {x, scale}, at::DeviceType::CUDA);

    torch::checkSameGPU("sym_quant", {x, "x", 0}, {scale, "scale", 1});
    torch::checkSize("sym_quant", torch::TensorArg{scale, "scale", 1}, 0, x.size(0));
    uint32_t rows = x.size(0);
    uint32_t colsSrc = x.size(1);
    uint32_t colsDst = cdiv(colsSrc, kElementsPerVector);

    auto q = torch::empty({rows, colsDst},torch::dtype(torch::kUInt8).device(x.device()));

    sym_quant_host((half*)x.data_ptr(), (half*)scale.data_ptr(), rows, colsSrc, colsDst, q.data_ptr<Int4Storage>());

    return q;
}

torch::Tensor sym_dual_quant(const torch::Tensor &x, const torch::Tensor &scale_1, const torch::Tensor &scale_2)
{
    torch::checkAllContiguous("sym_dual_quant", {{x,     "x",     0},
                                                      {scale_1, "scale_1", 1},
                                                      {scale_2, "scale_2", 2}});
    torch::checkDeviceType("sym_dual_quant", {x, scale_1, scale_2}, at::DeviceType::CUDA);

    torch::checkSameGPU("sym_dual_quant", {x, "x", 0}, {scale_1, "scale_1", 1});
    torch::checkSameGPU("sym_dual_quant", {x, "x", 0}, {scale_2, "scale_2", 2});
    torch::checkSize("sym_dual_quant", torch::TensorArg{scale_1, "scale_1", 1}, 0, x.size(0));
    torch::checkSize("sym_dual_quant", torch::TensorArg{scale_2, "scale_2", 2}, 0, x.size(0));
    uint32_t rows = x.size(0);
    uint32_t colsSrc = x.size(1);
    uint32_t colsDst = cdiv(colsSrc, kElementsPerVector);

    auto q = torch::empty({rows, colsDst},torch::dtype(torch::kUInt8).device(x.device()));

    sym_dual_quant_host((half*)x.data_ptr(), (half*)scale_1.data_ptr(), (half*)scale_2.data_ptr(), rows, colsSrc, colsDst, q.data_ptr<Int4Storage>());

    return q;
}


torch::Tensor sym_dequant(const torch::Tensor &q,
                                     const torch::Tensor &scale_row,
                                     const torch::Tensor &scale_col,
                                     const int bits)
{
    torch::checkAllContiguous("sym_dequant",
                              {{q,         "q",         0},
                               {scale_row, "scale_row", 1},
                               {scale_col, "scale_col", 2}
                              });
    torch::checkDeviceType("sym_dequant", {q, scale_row, scale_col},
                           at::DeviceType::CUDA);

    torch::checkAllSameGPU("sym_dequant",
                           {{q,         "q",         0},
                            {scale_row, "scale_row", 1},
                            {scale_col, "scale_col", 2}
                           });

    uint32_t rows = q.size(0);
    uint32_t cols = q.size(1);

    torch::checkSize("sym_dequant", torch::TensorArg{scale_row, "scale_row", 1}, 0,
                     rows);
    torch::checkSize("sym_dequant", torch::TensorArg{scale_col, "scale_col", 2}, 0,
                     cols);

    auto x = torch::empty(q.sizes(), torch::dtype(torch::kHalf).device(q.device()));

    switch (bits)
    {
        case 32:
            sym_dequant_host(q.data_ptr<int32_t>(), (half*)scale_row.data_ptr(), (half*)scale_col.data_ptr(),
                    rows, cols, (half*)x.data_ptr());
            break;
        default:
            TORCH_CHECK(false, "Unsupported data type")
    }

    return x;
}

torch::Tensor sym_dual_dequant(const torch::Tensor &q,
                                     const torch::Tensor &scale_row,
                                     const torch::Tensor &scale_col_1,
                                     const torch::Tensor &scale_col_2,
                                     const int bits)
{
    torch::checkAllContiguous("sym_dual_dequant",
                              {{q,         "q",         0},
                               {scale_row, "scale_row", 1},
                               {scale_col_1, "scale_col_1", 2},
                               {scale_col_2, "scale_col_2", 3}
                              });
    torch::checkDeviceType("sym_dual_dequant", {q, scale_row, scale_col_1, scale_col_2},
                           at::DeviceType::CUDA);

    torch::checkAllSameGPU("sym_dual_dequant",
                           {{q,         "q",         0},
                            {scale_row, "scale_row", 1},
                            {scale_col_1, "scale_col_1", 2},
                            {scale_col_2, "scale_col_2", 3}
                           });

    uint32_t rows = q.size(0);
    uint32_t cols = q.size(1);

    torch::checkSize("sym_dual_dequant", torch::TensorArg{scale_row, "scale_row", 1}, 0,
                     rows);
    torch::checkSize("sym_dual_dequant", torch::TensorArg{scale_col_1, "scale_col_1", 2}, 0,
                     cols);
    torch::checkSize("sym_dual_dequant", torch::TensorArg{scale_col_2, "scale_col_2", 3}, 0,
                     cols);

    auto x = torch::empty(q.sizes(), torch::dtype(torch::kHalf).device(q.device()));

    switch (bits)
    {
        case 32:
            sym_dual_dequant_host(q.data_ptr<int32_t>(), (half*)scale_row.data_ptr(), (half*)scale_col_1.data_ptr(), (half*)scale_col_2.data_ptr(),
                    rows, cols, (half*)x.data_ptr());
            break;
        default:
            TORCH_CHECK(false, "Unsupported data type")
    }

    return x;
}

// ===== Asym Quant/Dequant ======

torch::Tensor asym_quant(
    const torch::Tensor &x,
    const torch::Tensor &scale, const torch::Tensor &zeros
)
{
    torch::checkAllContiguous("asym_quant", {{x,     "x",     0},
                                             {scale, "scale", 1}, {zeros, "zeros", 2}});
    torch::checkDeviceType("asym_quant", {x, scale, zeros}, at::DeviceType::CUDA);
    torch::checkSameGPU("asym_quant", {x, "x", 0}, {scale, "scale", 1});
    torch::checkSameGPU("asym_quant", {x, "x", 0}, {zeros, "zeros", 2});
    torch::checkSize("asym_quant", torch::TensorArg{scale, "scale", 1}, 0, x.size(0));
    torch::checkSize("asym_quant", torch::TensorArg{zeros, "zeros", 2}, 0, x.size(0));

    uint32_t rows = x.size(0);
    uint32_t colsSrc = x.size(1);
    uint32_t colsDst = cdiv(colsSrc, kElementsPerVectorUInt4);

    auto q = torch::empty({rows, colsDst}, torch::dtype(torch::kUInt8).device(x.device()));

    asym_quant_host((half*)x.data_ptr(), (half*)scale.data_ptr(), (half*)zeros.data_ptr(), rows, colsSrc, colsDst, q.data_ptr<UInt4Storage>());
    
    return q;
}

torch::Tensor asym_dequant(
    const torch::Tensor &q,
    const torch::Tensor &scale_row, const torch::Tensor &zeros_row,
    const torch::Tensor &scale_col, const torch::Tensor &zeros_col,
    const int bits
)
{
    torch::checkAllContiguous("asym_dequant", {{q,       "q",        0},
                              {scale_row, "scale_row", 1}, {zeros_row, "zeros_row", 2},
                              {scale_col, "scale_col", 3}, {zeros_col, "zeros_col", 4}});
    torch::checkDeviceType("asym_dequant", {q, scale_row, zeros_row, scale_col, zeros_col}, 
                            at::DeviceType::CUDA);
    torch::checkAllSameGPU("asym_dequant", {{q, "q", 0}, 
                         {scale_row, "scale_row", 1}, {zeros_row, "zeros_row", 2},
                         {scale_col, "scale_col", 3}, {zeros_col, "zeros_col", 4}});
    
    uint32_t rows = q.size(0);
    uint32_t cols = q.size(1);

    torch::checkSize("asym_dequant", torch::TensorArg{scale_row, "scale_row", 1}, 0, rows);
    torch::checkSize("asym_dequant", torch::TensorArg{zeros_row, "zeros_row", 2}, 0, rows);
    torch::checkSize("asym_dequant", torch::TensorArg{scale_col, "scale_col", 3}, 0, cols);
    torch::checkSize("asym_dequant", torch::TensorArg{zeros_col, "zeros_col", 4}, 0, cols);

    auto x = torch::empty(q.sizes(), torch::dtype(torch::kHalf).device(q.device()));

    switch (bits)
    {
        case 32:
            asym_dequant_host(q.data_ptr<int32_t>(), 
                              (half*)scale_row.data_ptr(), (half*)zeros_row.data_ptr(),
                              (half*)scale_col.data_ptr(), (half*)zeros_col.data_ptr(),
                              rows, cols, (half*)x.data_ptr());
            break;
        default:
            TORCH_CHECK(false, "Unsupported data type")
    }

    return x;
}

torch::Tensor asym_dual_quant(
    const torch::Tensor &x,
    const torch::Tensor &scale_1, const torch::Tensor &zeros_1,
    const torch::Tensor &scale_2, const torch::Tensor &zeros_2
)
{
    torch::checkAllContiguous("asym_dual_quant", {{x,     "x",     0},
                                                   {scale_1, "scale_1", 1}, {zeros_1, "zeros_1", 2},
                                                   {scale_2, "scale_2", 3}, {zeros_2, "zeros_2", 4}});
    torch::checkDeviceType("asym_dual_quant", {x, scale_1, zeros_1, scale_2, zeros_2}, at::DeviceType::CUDA);
    torch::checkSameGPU("asym_dual_quant", {x, "x", 0}, {scale_1, "scale_1", 1});
    torch::checkSameGPU("asym_dual_quant", {x, "x", 0}, {zeros_1, "zeros_1", 2});
    torch::checkSameGPU("asym_dual_quant", {x, "x", 0}, {scale_2, "scale_2", 3});
    torch::checkSameGPU("asym_dual_quant", {x, "x", 0}, {zeros_2, "zeros_2", 4});
    torch::checkSize("asym_dual_quant", torch::TensorArg{scale_1, "scale_1", 1}, 0, x.size(0));
    torch::checkSize("asym_dual_quant", torch::TensorArg{zeros_1, "zeros_1", 2}, 0, x.size(0));
    torch::checkSize("asym_dual_quant", torch::TensorArg{scale_2, "scale_2", 3}, 0, x.size(0));
    torch::checkSize("asym_dual_quant", torch::TensorArg{zeros_2, "zeros_2", 4}, 0, x.size(0));

    uint32_t rows = x.size(0);
    uint32_t colsSrc = x.size(1);
    uint32_t colsDst = cdiv(colsSrc, kElementsPerVectorUInt4);

    auto q = torch::empty({rows, colsDst}, torch::dtype(torch::kUInt8).device(x.device()));
    asym_dual_quant_host((half*)x.data_ptr(), 
                         (half*)scale_1.data_ptr(), (half*)zeros_1.data_ptr(),
                         (half*)scale_2.data_ptr(), (half*)zeros_2.data_ptr(), 
                         rows, colsSrc, colsDst, q.data_ptr<UInt4Storage>());
}

torch::Tensor asym_dual_dequant(
    const torch::Tensor &q,
    const torch::Tensor &scale_row, const torch::Tensor &zeros_row,
    const torch::Tensor &scale_col_1, const torch::Tensor &zeros_col_1,
    const torch::Tensor &scale_col_2, const torch::Tensor &zeros_col_2,
    const int bits
)
{
    torch::checkAllContiguous("asym_dual_dequant", {{q,       "q",        0},
                              {scale_row, "scale_row", 1}, {zeros_row, "zeros_row", 2},
                              {scale_col_1, "scale_col_1", 3}, {zeros_col_1, "zeros_col_1", 4},
                              {scale_col_2, "scale_col_2", 5}, {zeros_col_2, "zeros_col_2", 6}});
   
    torch::checkDeviceType("asym_dual_dequant", {q, scale_row, zeros_row, scale_col_1, zeros_col_1, scale_col_2, zeros_col_2}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("asym_dual_dequant", {{q, "q", 0}, 
                         {scale_row, "scale_row", 1}, {zeros_row, "zeros_row", 2},
                         {scale_col_1, "scale_col_1", 3}, {zeros_col_1, "zeros_col_1", 4},
                         {scale_col_2, "scale_col_2", 5}, {zeros_col_2, "zeros_col_2", 6}});
    
    uint32_t rows = q.size(0);
    uint32_t cols = q.size(1);
    torch::checkSize("asym_dual_dequant", torch::TensorArg{scale_row, "scale_row", 1}, 0, rows);
    torch::checkSize("asym_dual_dequant", torch::TensorArg{zeros_row, "zeros_row", 2}, 0, rows);
    torch::checkSize("asym_dual_dequant", torch::TensorArg{scale_col_1, "scale_col_1", 3}, 0, cols);
    torch::checkSize("asym_dual_dequant", torch::TensorArg{zeros_col_1, "zeros_col_1", 4}, 0, cols);
    torch::checkSize("asym_dual_dequant", torch::TensorArg{scale_col_2, "scale_col_2", 5}, 0, cols);
    torch::checkSize("asym_dual_dequant", torch::TensorArg{zeros_col_2, "zeros_col_2", 6}, 0, cols);

    auto x = torch::empty(q.sizes(), torch::dtype(torch::kHalf).device(q.device()));

    switch (bits)
    {
        case 32:
            asym_dual_dequant_host(q.data_ptr<int32_t>(), 
                                   (half*)scale_row.data_ptr(), (half*)zeros_row.data_ptr(),
                                   (half*)scale_col_1.data_ptr(), (half*)zeros_col_1.data_ptr(),
                                   (half*)scale_col_2.data_ptr(), (half*)zeros_col_2.data_ptr(),
                                   rows, cols, (half*)x.data_ptr());
            break;
        default:
            TORCH_CHECK(false, "Unsupported data type")
    }

}

// ===== Flash Infer ======
inline void check_shape(const torch::Tensor &a, const torch::Tensor &b,
                        const char *a_name, const char *b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) \
  TORCH_CHECK(a == b, "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)


void batch_decode_i4(torch::Tensor o, torch::Tensor q, torch::Tensor kv_data,
                     torch::Tensor kv_param, torch::Tensor kv_indptr,
                     torch::Tensor kv_indicies, torch::Tensor last_page_offset,
                     int layer_idx) {
  CHECK_INPUT(o);
  CHECK_INPUT(q);
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);

  CHECK_DIM(3, o);                 // [B, N, D]
  CHECK_DIM(3, q);                 // [B, N, D]
  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 2]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Byte);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5)) * 2;
  int batch_size = static_cast<int>(o.size(0));
  CHECK_SHAPE(o, q);
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_EQ(head_dim, 128);

  FlashInferBatchDecodeKernel_i4<128>(
      (nv_half *)o.data_ptr(), (nv_half *)q.data_ptr(),
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), num_layers, layer_idx, num_heads,
      page_size, batch_size);
}

void init_kv_i4(torch::Tensor kv_data, torch::Tensor kv_param,
                torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                torch::Tensor last_page_offset, torch::Tensor k,
                torch::Tensor v, torch::Tensor k_param, torch::Tensor v_param,
                torch::Tensor seqlen_indptr, int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(seqlen_indptr);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 1]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, v);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, k_param);           // [sum(seqlen_i), N, 1]
  CHECK_DIM(3, v_param);           // [sum(seqlen_i), N, 1]
  CHECK_DIM(1, seqlen_indptr);     // [B+1]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Byte);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5)) * 2;
  int batch_size = static_cast<int>(last_page_offset.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(seqlen_indptr.size(0), batch_size + 1);
  CHECK_EQ(head_dim, 128);

  FlashInferInitKvKernel_i4<128>(
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), (void *)k.data_ptr(),
      (void *)v.data_ptr(), (nv_half2 *)k_param.data_ptr(),
      (nv_half2 *)v_param.data_ptr(), seqlen_indptr.data_ptr<int32_t>(),
      num_layers, layer_idx, num_heads, page_size, batch_size);
}

void append_kv_i4(torch::Tensor kv_data, torch::Tensor kv_param,
                  torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                  torch::Tensor last_page_offset, torch::Tensor k,
                  torch::Tensor v, torch::Tensor k_param, torch::Tensor v_param,
                  int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 1]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [B, N, D]
  CHECK_DIM(3, v);                 // [B, N, D]
  CHECK_DIM(3, k_param);           // [B, N, 1]
  CHECK_DIM(3, v_param);           // [B, N, 1]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Byte);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5)) * 2;
  int batch_size = static_cast<int>(k.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_SHAPE(k, v);
  CHECK_EQ(head_dim, 128);

  FlashInferAppendKvKernel_i4<128>(
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), (void *)k.data_ptr(),
      (void *)v.data_ptr(), (nv_half2 *)k_param.data_ptr(),
      (nv_half2 *)v_param.data_ptr(), num_layers, layer_idx, num_heads,
      page_size, batch_size);
}

void batch_decode_f16(torch::Tensor o, torch::Tensor q, torch::Tensor kv_data,
                     torch::Tensor kv_param, torch::Tensor kv_indptr,
                     torch::Tensor kv_indicies, torch::Tensor last_page_offset,
                     int layer_idx) {
  CHECK_INPUT(o);
  CHECK_INPUT(q);
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);

  CHECK_DIM(3, o);                 // [B, N, D]
  CHECK_DIM(3, q);                 // [B, N, D]
  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 2]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Half);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(o.size(0));
  CHECK_SHAPE(o, q);
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_EQ(head_dim, 128);

  FlashInferBatchDecodeKernel_f16<128>(
      (nv_half *)o.data_ptr(), (nv_half *)q.data_ptr(),
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), num_layers, layer_idx, num_heads,
      page_size, batch_size);
}

void init_kv_f16(torch::Tensor kv_data, torch::Tensor kv_param,
                torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                torch::Tensor last_page_offset, torch::Tensor k,
                torch::Tensor v, torch::Tensor k_param, torch::Tensor v_param,
                torch::Tensor seqlen_indptr, int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(seqlen_indptr);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 1]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, v);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, k_param);           // [sum(seqlen_i), N, 1]
  CHECK_DIM(3, v_param);           // [sum(seqlen_i), N, 1]
  CHECK_DIM(1, seqlen_indptr);     // [B+1]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Half);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(last_page_offset.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(seqlen_indptr.size(0), batch_size + 1);
  CHECK_EQ(head_dim, 128);

  FlashInferInitKvKernel_f16<128>(
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), (void *)k.data_ptr(),
      (void *)v.data_ptr(), (nv_half2 *)k_param.data_ptr(),
      (nv_half2 *)v_param.data_ptr(), seqlen_indptr.data_ptr<int32_t>(),
      num_layers, layer_idx, num_heads, page_size, batch_size);
}

void append_kv_f16(torch::Tensor kv_data, torch::Tensor kv_param,
                  torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                  torch::Tensor last_page_offset, torch::Tensor k,
                  torch::Tensor v, torch::Tensor k_param, torch::Tensor v_param,
                  int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(6, kv_param);          // [None, L, 2, N, P, 1]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [B, N, D]
  CHECK_DIM(3, v);                 // [B, N, D]
  CHECK_DIM(3, k_param);           // [B, N, 1]
  CHECK_DIM(3, v_param);           // [B, N, 1]

  CHECK_EQ(kv_data.scalar_type(), at::ScalarType::Half);
  CHECK_EQ(kv_param.scalar_type(), at::ScalarType::Half);

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(k.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_SHAPE(k, v);
  CHECK_EQ(head_dim, 128);

  FlashInferAppendKvKernel_f16<128>(
      (void *)kv_data.data_ptr(), (nv_half2 *)kv_param.data_ptr(),
      kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),
      last_page_offset.data_ptr<int32_t>(), (void *)k.data_ptr(),
      (void *)v.data_ptr(), (nv_half2 *)k_param.data_ptr(),
      (nv_half2 *)v_param.data_ptr(), num_layers, layer_idx, num_heads,
      page_size, batch_size);
}



//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
)
{

    m.def("matmul", &matmul,
          "input: (A: torch.Tensor(M x K, UINT8, CUDA), B: torch.Tensor(N x K, "
          "UINT8, CUDA))\n"
          "output: torch.Tensor(M x N, INT32, CUDA)\n"
          "output = int4Unpacking(A) @ int4Unpacking(B)^T",
          py::arg("A"), py::arg("B"));

    m.def("sym_quant", &sym_quant,
          "input: (src: torch.Tensor(M x N, FP16, CUDA), scale: "
          "torch.Tensor(M x 1, FP16, CUDA))"
          "bits: int\n"
          "output: torch.Tensor(M x ceil(N / 2), UINT8, CUDA)\n"
          "output = int4Packing(int4Rounding(source / scale)\n",
          py::arg("x"), py::arg("scale"));

    m.def("sym_dual_quant", &sym_dual_quant,
            "input: (src: torch.Tensor(M x N, FP16, CUDA), scale_1: "
            "torch.Tensor(M x 1, FP16, CUDA), scale_2: torch.Tensor(M x 1, FP16, CUDA))"
            "bits: int\n"
            "output: torch.Tensor(M x ceil(N / 2), UINT8, CUDA)\n"
            "output = int4Packing(int4Rounding(source / scale_1 / scale_2)\n",
            py::arg("x"), py::arg("scale_1"), py::arg("scale_2"));

    m.def("sym_dequant", &sym_dequant,
          "input (x: torch.Tensor(M x N), scale_row: torch.Tensor(M x 1, "
          "FP16), scale_col: torch.Tensor(1 x N, FP16)"
          "bits: int\n"
          "output: torch.Tensor(M x N, FP16)\n"
          "output = x * scale_row * scale_col"
          "when bits equal 8: "
          "input x type is int8\n"
          "when bits equal 16: "
          "input x type is FP16\n"
          "when bits equal 32: "
          "input x type is int32\n",
          py::arg("q"), py::arg("scale_row"), py::arg("scale_col"),
          py::arg("bits"));

    m.def("sym_dual_dequant", &sym_dual_dequant,
            "input (x: torch.Tensor(M x N), scale_row: torch.Tensor(M x 1, "
            "FP16), scale_col_1: torch.Tensor(1 x N, FP16), scale_col_2: torch.Tensor(1 x N, FP16)"
            "bits: int\n"
            "output: torch.Tensor(M x N, FP16)\n"
            "output = x * scale_row * scale_col_1 * scale_col_2"
            "when bits equal 8: "
            "input x type is int8\n"
            "when bits equal 16: "
            "input x type is FP16\n"
            "when bits equal 32: "
            "input x type is int32\n",
            py::arg("q"), py::arg("scale_row"), py::arg("scale_col_1"), py::arg("scale_col_2"),
            py::arg("bits"));

    m.def("asym_quant", &asym_quant,
        "input: (src: torch.Tensor(M x N, FP16, CUDA), scale: "
        "torch.Tensor(M x 1, FP16, CUDA), zeros: torch.Tensor(M x 1, FP16, CUDA))"
        "output: torch.Tensor(M x ceil(N / 2), UINT8, CUDA)\n"
        "output = int4Packing(int4Rounding(source / scale)\n",
        py::arg("x"), py::arg("scale"), py::arg("zeros"));

    m.def("asym_dequant", &asym_dequant,
        "input (x: torch.Tensor(M x N), scale_row: torch.Tensor(M x 1, "
        "FP16), zeros_row: torch::Tensor(M x 1, FP16), scale_col: torch::Tensor(1 x N, FP16), zeros_col: torch::Tensor(1 x N, FP16)"
        "bits: int\n"
        "output: torch.Tensor(M x N, FP16)\n"
        "output = x * scale_row * scale_col"
        "when bits equal 8: "
        "input x type is int8\n"
        "when bits equal 16: "
        "input x type is FP16\n"
        "when bits equal 32: "
        "input x type is int32\n",
        py::arg("q"), py::arg("scale_row"), py::arg("zeros_row"), py::arg("scale_col"), py::arg("zeros_col"),
        py::arg("bits"));
    
    m.def("asym_dual_quant", &asym_dual_quant, 
        "input: (src: torch.Tensor(M x N, FP16, CUDA), scale_1: "
        "torch.Tensor(M x 1, FP16, CUDA), zeros_1: torch.Tensor(M x 1, FP16, CUDA), "
        "scale_2: torch.Tensor(M x 1, FP16, CUDA), zeros_2: torch.Tensor(M x 1, FP16, CUDA))"
        "output: torch.Tensor(M x ceil(N / 2), UINT8, CUDA)\n"
        "output = int4Packing(int4Rounding(source / scale_1 / scale_2)\n",
        py::arg("x"), py::arg("scale_1"), py::arg("zeros_1"), py::arg("scale_2"), py::arg("zeros_2"));
    
    m.def("asym_dual_dequant", &asym_dual_dequant,
        "input (x: torch.Tensor(M x N), scale_row: torch.Tensor(M x 1, "
        "FP16), zeros_row: torch::Tensor(M x 1, FP16), scale_col_1: torch::Tensor(1 x N, FP16), zeros_col_1: torch::Tensor(1 x N, FP16), "
        "scale_col_2: torch::Tensor(1 x N, FP16), zeros_col_2: torch::Tensor(1 x N, FP16)"
        "bits: int\n"
        "output: torch.Tensor(M x N, FP16)\n"
        "output = x * scale_row * scale_col_1 * scale_col_2"
        "when bits equal 8: "
        "input x type is int8\n"
        "when bits equal 16: "
        "input x type is FP16\n"
        "when bits equal 32: "
        "input x type is int32\n",
        py::arg("q"), py::arg("scale_row"), py::arg("zeros_row"), py::arg("scale_col_1"), py::arg("zeros_col_1"), py::arg("scale_col_2"), py::arg("zeros_col_2"),
        py::arg("bits"));

    m.def("batch_decode_i4", &batch_decode_i4, "");
    m.def("init_kv_i4", &init_kv_i4, "");
    m.def("append_kv_i4", &append_kv_i4, "");
    m.def("batch_decode_f16", &batch_decode_f16, "");
    m.def("init_kv_f16", &init_kv_f16, "");
    m.def("append_kv_f16", &append_kv_f16, ""); 

}
