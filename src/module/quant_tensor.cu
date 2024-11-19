#include <quant_tensor.h>
#include <ATen/TensorUtils.h>

#include <iostream>

inline uint32_t cdiv(uint32_t x, uint32_t y)
{
    return (x + y - 1) / y;
}

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

torch::Tensor sym_double_quant(const torch::Tensor &x, const torch::Tensor &scale_1, const torch::Tensor &scale_2)
{
    torch::checkAllContiguous("sym_quant", {{x,       "x",       0},
                                            {scale_1, "scale_1", 1},
                                            {scale_2, "scale_2", 2}});
    torch::checkDeviceType("sym_quant", {x, scale_1, scale_2}, at::DeviceType::CUDA);
    torch::checkSameGPU("sym_quant", {x, "x", 0}, {scale_1, "scale_1", 1});
    torch::checkSameGPU("sym_quant", {x, "x", 0}, {scale_2, "scale_2", 2});
    torch::checkSize("sym_quant", torch::TensorArg{scale_1, "scale_1", 1}, 0, x.size(0));
    torch::checkSize("sym_quant", torch::TensorArg{scale_2, "scale_2", 2}, 0, x.size(0));

    uint32_t rows = x.size(0);
    uint32_t colsSrc = x.size(1);
    uint32_t colsDst = cdiv(colsSrc, kElementsPerVector);

    auto q = torch::empty({rows, colsDst}, torch::dtype(torch::kUInt8).device(x.device()));

    sym_double_quant_host(/* x       */(half*)x.data_ptr(), 
                          /* scale_1 */(half*)scale_1.data_ptr(),
                          /* scale_2 */(half*)scale_2.data_ptr(),
                          /* rows    */rows, 
                          /* colsSrc */colsSrc, 
                          /* colsDst */colsDst, 
                          /* q       */q.data_ptr<Int4Storage>());

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

torch::Tensor sym_double_dequant(const torch::Tensor &q,
                                 const torch::Tensor &scale_row,
                                 const torch::Tensor &scale_col_1,
                                 const torch::Tensor &scale_col_2,
                                 const int bits)
{
    torch::checkAllContiguous("sym_double_dequant",
                              {{q,           "q",           0},
                               {scale_row,   "scale_row",   1},
                               {scale_col_1, "scale_col_1", 2},
                               {scale_col_2, "scale_col_2", 3}
                              });
    torch::checkDeviceType("sym_double_dequant", {q, scale_row, scale_col_1, scale_col_2},
                            at::DeviceType::CUDA);
    torch::checkAllSameGPU("sym_double_dequant",
                           {{q,           "q",           0},
                            {scale_row,   "scale_row",   1},
                            {scale_col_1, "scale_col_1", 2},
                            {scale_col_2, "scale_col_2", 3}
                           });

    uint32_t rows = q.size(0);
    uint32_t cols = q.size(1);

    torch::checkSize("sym_double_dequant", torch::TensorArg{scale_row, "scale_row", 1}, 0,
                     rows);
    torch::checkSize("sym_double_dequant", torch::TensorArg{scale_col_1, "scale_col_1", 2}, 0,
                     cols);
    torch::checkSize("sym_double_dequant", torch::TensorArg{scale_col_2, "scale_col_2", 3}, 0,
                     cols);
    auto x = torch::empty(q.sizes(), torch::dtype(torch::kHalf).device(q.device()));

    switch (bits)
    {
        case 32:
            sym_double_dequant_host(q.data_ptr<int32_t>(), 
                                    (half*)scale_row.data_ptr(), 
                                    (half*)scale_col_1.data_ptr(),
                                    (half*)scale_col_2.data_ptr(), 
                                    rows, cols, 
                                    (half*)x.data_ptr());
            break;
        default:
            TORCH_CHECK(false, "Unsupported data type")
    }

    return x;
}


PackedQuantizedTensor Quantizer4bit::forward(torch::Tensor x)
{
    auto abs_x = torch::abs(x);
    auto max_abs_x = std::get<0>(torch::max(abs_x, /*dim=*/-1));
    auto scales_x = (max_abs_x.unsqueeze(1) / 7.0).to(torch::kFloat16) * input_clip_ratio;

    auto quantized_x = sym_quant(x, scales_x);
    // sym_quant
    PackedQuantizedTensor packed_tensor(quantized_x, scales_x);
    return packed_tensor;
}

// torch::Tensor DeQuantizer4bit::forward(PackedQuantizedTensor packed_tensor, torch::Tensor weight_scale)
// {
//     auto x = packed_tensor.quantized_x;
//     auto scale_x = packed_tensor.scale;

//     auto dequantized_x = sym_dequant(x, scale_x, weight_scale, 32);
//     return dequantized_x;
// }

torch::Tensor DeQuantizer4bit::forward(torch::Tensor &x, torch::Tensor &scale_x, torch::Tensor &weight_scale)
{
    auto dequantized_x = sym_dequant(x, scale_x, weight_scale, 32);
    return dequantized_x;
}

std::pair<torch::Tensor, torch::Tensor> get_minq_maxq(int bits, bool sym)
{
    torch::Tensor maxq, minq;
    if (sym) {
        maxq = torch::tensor((1 << (bits - 1)) - 1);
        minq = torch::tensor(-(1 << (bits - 1)));
    } else {
        maxq = torch::tensor((1 << bits) - 1);
        minq = torch::tensor(0);
    }
    return std::make_pair(minq, maxq);
}

torch::Tensor two_compl(torch::Tensor x, int bits)
{
    auto mask = (1 << bits);
    return torch::where(x < 0, mask + x, x);
}

torch::Tensor pack_i4(torch::Tensor q)
{
    assert(q.scalar_type() == torch::kInt8 || q.scalar_type() == torch::kInt16 || q.scalar_type() == torch::kInt32 || q.scalar_type() == torch::kInt64);
    auto[minq, maxq] = get_minq_maxq(4, true);
    assert(torch::all(q.ge(minq) & q.le(maxq)).item<bool>());

    auto q_i8 = two_compl(q.to(torch::kInt8), 4).to(torch::kUInt8);
    auto q_i4 = q_i8.slice(1, 0, -1, 2) | (q_i8.slice(1, 1, torch::nullopt, 2) * 16);
    return q_i4;
}
