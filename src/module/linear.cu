#include <linear.h>

std::pair<torch::Tensor, std::vector<int64_t>> flatten_last_dim_and_return_shape(torch::Tensor &x)
{
    std::vector<int64_t> shape_excl_last(x.sizes().begin(), x.sizes().end() - 1);
    x = x.view({-1, x.size(-1)});
    return std::make_pair(x, shape_excl_last);
}

torch::Tensor reshape_tensor(torch::Tensor tensor, const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2)
{
    std::vector<int64_t> combined_shape = shape1;
    combined_shape.insert(combined_shape.end(), shape2.begin(), shape2.end());
    tensor = tensor.view(combined_shape);
    return tensor;
}

torch::Tensor reshape_tensor(torch::Tensor tensor, const std::vector<int64_t> &shape, const int shape2)
{
    std::vector<int64_t> combined_shape = shape;
    combined_shape.push_back(shape2);
    tensor = tensor.view(combined_shape);
    return tensor;
}

Linear::Linear(uint32_t input_dim, uint32_t output_dim, bool bias, torch::Dtype dtype)
{
    this->weight = torch::randn({output_dim, input_dim}, dtype);
    this->weight.set_requires_grad(false);
    this->weight = this->weight.to(torch::kCUDA);

    if (bias)
    {
        this->bias = torch::zeros({output_dim}, dtype);
        this->bias.set_requires_grad(false);
        this->bias = this->bias.to(torch::kCUDA);
    }
}

torch::Tensor Linear::forward(torch::Tensor input)
{
    auto [A, A_shape_excl_last] = flatten_last_dim_and_return_shape(input);
    auto [B, B_shape_excl_last] = flatten_last_dim_and_return_shape(this->weight);

    torch::checkAllContiguous("Linear::forward", {{A, "A", 0},
                                                  {B, "B", 1}});
    torch::checkDeviceType("Linear::forward", {A, B}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("Linear::forward", {{A, "A", 0},
                                               {B, "B", 1}});
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    uint32_t K = A.size(1);

    torch::Tensor C = torch::empty({M, N}, torch::dtype(this->weight.dtype()).device(A.device()));

    if (A.dtype() != this->weight.dtype())
    {
        LOG_WARN("Find different dtype between input and weight");
        A = A.to(this->weight.dtype());
    }

    matmul_host<half>(
        (half *)A.data_ptr(),
        (half *)B.data_ptr(),
        M, N, K,
        (half *)C.data_ptr()
    );

    // matmul_host<float>(
    //     (float *)A.data_ptr(),
    //     (float *)B.data_ptr(),
    //     M, N, K,
    //     (float *)C.data_ptr()
    // );
    
    C = reshape_tensor(C, A_shape_excl_last, B_shape_excl_last);

    if (this->bias.defined())
    {
        C = C + this->bias;
    }

    return C;
}


Linear4bit::Linear4bit(uint32_t input_dim, uint32_t output_dim, bool bias, torch::Dtype dtype)
{
    this->weight = torch::randint(1, 7, {output_dim, input_dim / 2}, torch::kUInt8);
    this->weight_scale = torch::zeros({output_dim, 1}, torch::kFloat16);

    this->weight.set_requires_grad(false);
    this->weight_scale.set_requires_grad(false);

    this->weight = this->weight.to(torch::kCUDA);
    this->weight_scale = this->weight_scale.to(torch::kCUDA);

    if (bias)
    {
        this->bias = torch::zeros({output_dim}, dtype);
        this->bias.set_requires_grad(false);
        this->bias = this->bias.to(torch::kCUDA);
    }
}

// Linear4bit::~Linear4bit() {}

torch::Tensor Linear4bit::forward(PackedQuantizedTensor input)
{
    torch::Tensor &x = input.quantized_x;
    torch::Tensor &scale_row = input.scale;
    torch::Tensor &scale_col = this->weight_scale;

    // 4bit matmul forward
    assert(x.size(-1) % 32 == 0);
    auto [A, A_shape_excl_last] = flatten_last_dim_and_return_shape(x);
    auto [B, B_shape_excl_last] = flatten_last_dim_and_return_shape(this->weight);
    // pre-check
    torch::checkAllContiguous("Linear4bit::forward", {{A, "A", 0},
                                                      {B, "B", 1}});
    torch::checkDeviceType("Linear4bit::forward", {A, B}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("Linear4bit::forward", {{A, "A", 0},
                                                   {B, "B", 1}});
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    uint32_t K = A.size(1) * kElementsPerVector;

    torch::Tensor C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

    matmul_host_4bit(
        A.data_ptr<Int4Storage>(),
        B.data_ptr<Int4Storage>(),
        M, N, K,
        C.data_ptr<int32_t>()
    );

    C = reshape_tensor(C, A_shape_excl_last, B_shape_excl_last);

    // dequantized to float
    assert(C.dtype() == torch::kInt32);
    assert(scale_row.dtype() == scale_col.dtype() && scale_row.dtype() == torch::kFloat16);
    auto [res, res_shape_excl_last] = flatten_last_dim_and_return_shape(C);

    res = sym_dequant(res, scale_row.view(-1), scale_col, 32);
    res = reshape_tensor(res, res_shape_excl_last, -1);

    if (this->bias.defined())
    {
        res = res + this->bias;
    }

    return res;
}

Linear4bit Linear4bit::from_float(Linear &linear, torch::Tensor weight_scales)
{
    TORCH_CHECK(false, "Unimplemented function, current only support convert from torch::nn::Linear")
    //TODO: implement from_float(based on custom linear layer)
}

Linear4bit Linear4bit::from_float(torch::nn::Linear &linear, torch::Tensor weight_scales)
{
    auto weight_matrix = linear->weight.data();

    auto int_module = Linear4bit(linear->weight.size(-1), linear->weight.size(0), linear->bias.defined(), torch::kFloat16);   
    weight_matrix = weight_matrix.to(torch::kCUDA);
    int_module.weight_scale.copy_(weight_scales.to(torch::kCUDA, weight_matrix.dtype()));
    auto int_rounded_weight = torch::round(weight_matrix / int_module.weight_scale);
    int_module.weight.copy_(pack_i4(int_rounded_weight.to(torch::kInt8)));

    if (linear->bias.defined())
    {
        int_module.bias.copy_(linear->bias.data());
    }
    return int_module;
}

Linear4bitDUQ::Linear4bitDUQ(uint32_t input_dim, uint32_t output_dim, bool bias, torch::Dtype dtype)
{
    this->weight = torch::randint(1, 7, {output_dim, input_dim / 2}, torch::kUInt8);
    this->weight_scale_1 = torch::zeros({output_dim, 1}, torch::kFloat16);
    this->weight_scale_2 = torch::zeros({output_dim, 1}, torch::kFloat16);

    this->weight.set_requires_grad(false);
    this->weight_scale_1.set_requires_grad(false);
    this->weight_scale_2.set_requires_grad(false);

    this->weight = this->weight.to(torch::kCUDA);
    this->weight_scale_1 = this->weight_scale_1.to(torch::kCUDA);
    this->weight_scale_2 = this->weight_scale_2.to(torch::kCUDA);

    if (bias)
    {
        this->bias = torch::zeros({output_dim}, dtype);
        this->bias.set_requires_grad(false);
        this->bias = this->bias.to(torch::kCUDA);
    }
}

torch::Tensor Linear4bitDUQ::forward(PackedQuantizedTensor input)
{
    torch::Tensor &x = input.quantized_x;
    torch::Tensor &scale_row = input.scale;
    torch::Tensor &scale_col_1 = this->weight_scale_1;
    torch::Tensor &scale_col_2 = this->weight_scale_2;

    // 4bit matmul forward
    assert(x.size(-1) % 32 == 0);
    auto [A, A_shape_excl_last] = flatten_last_dim_and_return_shape(x);
    auto [B, B_shape_excl_last] = flatten_last_dim_and_return_shape(this->weight);
    // pre-check
    torch::checkAllContiguous("Linear4bitDUQ::forward", {{A, "A", 0},
                                                         {B, "B", 1}});
    torch::checkDeviceType("Linear4bitDUQ::forward", {A, B}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("Linear4bitDUQ::forward", {{A, "A", 0},
                                                      {B, "B", 1}});
    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    uint32_t K = A.size(1) * kElementsPerVector;

    torch::Tensor C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

    matmul_host_4bit(
        A.data_ptr<Int4Storage>(),
        B.data_ptr<Int4Storage>(),
        M, N, K,
        C.data_ptr<int32_t>()
    );

    C = reshape_tensor(C, A_shape_excl_last, B_shape_excl_last);

    // dequantized to float
    assert(C.dtype() == torch::kInt32);
    assert(scale_row.dtype() == scale_col_1.dtype() && scale_col_1.dtype() == scale_col_2.dtype() && scale_row.dtype() == torch::kFloat16);
    auto [res, res_shape_excl_last] = flatten_last_dim_and_return_shape(C);

    res = sym_double_dequant(res, scale_row.view(-1), scale_col_1, scale_col_2, 32);
    res = reshape_tensor(res, res_shape_excl_last, -1);

    if (this->bias.defined())
    {
        res = res + this->bias;
    }

    return res;
}

Linear4bitDUQ Linear4bitDUQ::from_float(Linear &linear, torch::Tensor weigh_scale_1, torch::Tensor weight_scale_2)
{
    TORCH_CHECK(false, "Unimplemented function, current only support convert from torch::nn::Linear")
    // TODO: implement from_float(based on custom linear layer)
}

Linear4bitDUQ Linear4bitDUQ::from_float(torch::nn::Linear &linear, torch::Tensor weight_scale_1, torch::Tensor weight_scale_2)
{
    auto weight_matrix = linear->weight.data();
    
    auto int_module = Linear4bitDUQ(linear->weight.size(-1), linear->weight.size(0), linear->bias.defined(), torch::kFloat16);
    weight_matrix = weight_matrix.to(torch::kCUDA);
    int_module.weight_scale_1.copy_(weight_scale_1.to(torch::kCUDA, weight_matrix.dtype()));
    int_module.weight_scale_2.copy_(weight_scale_2.to(torch::kCUDA, weight_matrix.dtype()));

    auto weight_part1 = weight_matrix.index({"...", at::indexing::Slice(at::indexing::None, QUANT_COL_K)}).div(int_module.weight_scale_1);
    auto weight_part2 = weight_matrix.index({"...", at::indexing::Slice(QUANT_COL_K, at::indexing::None)}).div(int_module.weight_scale_2);

    auto int_rounded_weight_1 = torch::round(weight_part1).to(torch::kInt8);
    auto int_rounded_weight_2 = torch::round(weight_part2).to(torch::kInt8);

    auto int_rounded_weight = torch::empty({linear->weight.size(0), linear->weight.size(1)}, torch::kInt8);
    int_rounded_weight.index_put_({"...", at::indexing::Slice(at::indexing::None, QUANT_COL_K)}, int_rounded_weight_1);
    int_rounded_weight.index_put_({"...", at::indexing::Slice(QUANT_COL_K, at::indexing::None)}, int_rounded_weight_2);

    int_module.weight.copy_(pack_i4(int_rounded_weight));

    if (linear->bias.defined())
    {
        int_module.bias.copy_(linear->bias.data());
    }

    return int_module;
}
