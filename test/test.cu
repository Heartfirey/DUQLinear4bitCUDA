#include <iostream>
#include <stdlib.h>
#include <string.h>

#include <curand.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#include <gemm.h>
#include <linear.h>
#include <cutools.cuh>

#include <quant_tensor.h>
#include <benchmark.h>
#include <logger.h>
#include <indicators/progress_spinner.hpp>
#include <cuda_runtime.h>


void check1()
{
    uint32_t shape[3] = {4, 4, 4};
    uint32_t dimension = 3;

    float *tensor = NewTensor<float>(shape, dimension);

    if (tensor == nullptr) {
        std::cout << "Tensor init failed!" << std::endl;
    }

    CHECK_EXEC(cudaFree(tensor));
    std::cout << "Check1 Passed" << std::endl;
}

void check2()
{
    float templateTensorOnCpu[4][4] = {
        {1.0, 2.0, 3.0, 4.0},
        {1.0, 2.0, 3.0, 4.0},
        {1.0, 2.0, 3.0, 4.0},
        {1.0, 2.0, 3.0, 4.0},
    };

    uint32_t shape_1[2] = {4, 4};
    uint32_t dimension_1 = 2;

    float *tensor_1 = NewTensorFromHost<float>((float *)templateTensorOnCpu, shape_1, dimension_1);

    if (tensor_1 == nullptr) {
        std::cout << "Tensor copy failed!" << std::endl;
    }

    CHECK_EXEC(cudaFree(tensor_1));
    std::cout << "Check2 Passed" << std::endl;
}

void check3()
{
    float matrixA[15] = {
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
    };

    float matrixB[12] = {
        1.0, 2.0, 1.0,
        1.0, 3.0, 1.0,
        1.0, 4.0, 1.0,
        1.0, 5.0, 1.0,
    };

    uint32_t matrixShapeA[2] = {5, 3};
    uint32_t matrixShapeB[2] = {4, 3};
    uint32_t matrixShapeC[2] = {5, 4};
    uint32_t dimension = 2;

    float *tensorA = NewTensorFromHost<float>((float *)matrixA, matrixShapeA, dimension);
    float *tensorB = NewTensorFromHost<float>((float *)matrixB, matrixShapeB, dimension);

    assert(tensorA != nullptr);
    assert(tensorB != nullptr);

    float *tensorC = NewTensor<float>(matrixShapeC, dimension);
    matmul_host<float>(tensorA, tensorB, 5, 4, 3, tensorC);

    float matrixC[20] = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    };
    cudaMemcpy(matrixC, tensorC, 5*4*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 4; j++) {
            std::cout << matrixC[i * 4 + j] << ' ';
        }
        std::cout << std::endl;
    }

    std::cout << "Check3 Passed" << std::endl;
}

void check4() 
{
    uint32_t input_dim = 3, output_dim = 4, batch_size=5;
    LinearPlain<float> linear_test(input_dim, output_dim, true);
    float matrixA[15] = {
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
    };
    uint32_t input_shape[2] = {batch_size, input_dim};
    uint32_t output_shape[2] = {batch_size, output_dim};
    float *input_tensor = NewTensorFromHost<float>((float *)matrixA, input_shape, (uint32_t)2);
    float *output_tensor = NewTensor<float>(output_shape, (uint32_t)2);

    float fake_weight_cpu[12] = {
        1.0, 2.0, 2.0,
        2.0, 3.0, 1.0,
        1.0, 4.0, 3.0,
        3.0, 5.0, 1.0,
    };

    cudaMemcpy(linear_test.weight, fake_weight_cpu, 12 * sizeof(float), cudaMemcpyHostToDevice);

    float fake_bias_cpu[4] = {1.0, 1.0, 1.0, 1.0};

    cudaMemcpy(linear_test.bias, fake_bias_cpu, 4 * sizeof(float), cudaMemcpyHostToDevice);

    CudaTimer timer;
    timer.start();
    linear_test.forward(input_tensor, output_tensor, batch_size);
    timer.stop();

    LOG_INFO("FP32 Forward Time: {}ms", timer.elapsed());
    float matrixC[20];
    memset(matrixC, -1, 20 * sizeof(float));
    cudaMemcpy(matrixC, output_tensor, 20 * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 4; j++) {
            std::cout << matrixC[i * 4 + j] << ' ';
        }
        std::cout << std::endl;
    }

    std::cout << "Check4 Passed" << std::endl;
}

void check5() 
{

    torch::Tensor test_data_x = torch::randn({5, 3});
    torch::Tensor test_scale_x = torch::randn({3});
    PackedQuantizedTensor test_packed_quantized(test_data_x, test_scale_x);
    std::cout << test_packed_quantized.device() << std::endl;
    std::cout << torch::max(test_data_x) << std::endl;

    std::cout << "Check5 Passed" << std::endl;
}

void check6()
{

    torch::Tensor test_a = torch::tensor(
        {{1.0, 2.0, 3.0},
         {1.0, 2.0, 3.0},
         {1.0, 2.0, 3.0},
         {1.0, 2.0, 3.0},
         {1.0, 2.0, 3.0}}, torch::dtype(torch::kFloat32)
    ).to(torch::kCUDA);

    torch::Tensor test_b = torch::tensor(
        {{1.0, 2.0, 2.0},
        {2.0, 3.0, 1.0},
        {1.0, 4.0, 3.0},
        {3.0, 5.0, 1.0}}, torch::dtype(torch::kFloat32)
    ).to(torch::kCUDA);

    torch::Tensor test_c = torch::zeros({5, 4}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);

    torch::Tensor test_bias = torch::tensor(
        {1.0, 1.0, 1.0, 1.0}, torch::dtype(torch::kFloat32)
    ).to(torch::kCUDA);

    uint32_t input_dim = 3, output_dim = 4, batch_size=5;
    LinearPlain<float> linear_test(input_dim, output_dim, true);

    linear_test.weight = test_b.data_ptr<float>();
    linear_test.bias = test_bias.data_ptr<float>();

    linear_test.forward(test_a.data_ptr<float>(), test_c.data_ptr<float>(), batch_size);

    std::cout << test_c << std::endl;

}


// temp test function
torch::Tensor quantize_bfloat16_to_int4(const torch::Tensor& input) {
    if (input.scalar_type() != torch::kBFloat16) {
        throw std::runtime_error("Input tensor must be of BFloat16 type.");
    }
    auto min_val = input.min().item<float>();
    auto max_val = input.max().item<float>();
    float scale = 7.0f / std::max(std::abs(min_val), std::abs(max_val));
    torch::Tensor scaled = input.to(torch::kFloat).mul(scale).round();
    scaled.clamp_(-8, 7);
    return scaled.to(torch::kInt8);
}

void check7()
{
    // quantization check
    torch::Tensor test_a = torch::rand({4, 64}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);
    test_a *= 5;
    test_a = test_a.to(torch::dtype(torch::kBFloat16));

    std::cout << "Before Quantization" << std::endl;
    std::cout << test_a << std::endl;
    std::cout << "==================================" << std::endl << std::endl;

    Quantizer4bit quantizer(1.0);
    DeQuantizer4bit dequantizer;

    auto packed_quantized_a = quantizer.forward(test_a);

    torch::Tensor col_scale = torch::zeros({32, 1}).to(torch::kCUDA);  

    std::cout << "After Quantization" << std::endl;
    std::cout << packed_quantized_a.quantized_x << std::endl;
    std::cout << "==================================" << std::endl << std::endl;

    std::cout << "Scale" << std::endl;
    std::cout << packed_quantized_a.scale << std::endl;

    // auto dequant_result = dequantizer.forward(packed_quantized_a.quantized_x, packed_quantized_a.scale, col_scale);
    // dequant_result = dequant_result.to(torch::dtype(torch::kBFloat16));

    // std::cout << "After Dequantization" << std::endl;
    // std::cout << dequant_result << std::endl;
    // std::cout << "==================================" << std::endl << std::endl;
}

void check8()
{
    const uint32_t input_dim = 64, output_dim = 32, batch_size = 4;
    torch::Tensor test_a = torch::rand({batch_size, input_dim}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);
    torch::Tensor test_b = torch::rand({output_dim, input_dim}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);
    torch::Tensor test_c = torch::zeros({batch_size, output_dim}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);

    // scale to 16 for testing
    test_a *= 4;
    test_b *= 4;
    test_a = test_a.to(torch::dtype(torch::kBFloat16));
    test_b = test_b.to(torch::dtype(torch::kBFloat16));
    test_c = test_c.to(torch::dtype(torch::kInt32));
    
    Quantizer4bit quantizer(1.0);

    auto packed_quantized_a = quantizer.forward(test_a);
    auto packed_quantized_b = quantizer.forward(test_b);

    torch::Tensor &A = packed_quantized_a.quantized_x;
    torch::Tensor &B = packed_quantized_b.quantized_x;

    torch::checkAllContiguous("check7", {{A, "A", 0},
                                         {B, "B", 1}});

    torch::checkDeviceType("check7", {A, B}, at::DeviceType::CUDA);
    torch::checkAllSameGPU("check7", {{A, "A", 0},
                                      {B, "B", 1}});

    uint32_t M = A.size(0);
    uint32_t N = B.size(0);
    uint32_t K = A.size(1) * 2;

    matmul_host_4bit(
        A.data_ptr<Int4Storage>(),
        B.data_ptr<Int4Storage>(),
        /*M*/ batch_size, 
        /*N*/ output_dim, 
        /*K*/ input_dim,
        test_c.data_ptr<int32_t>()
    );

    DeQuantizer4bit dequantizer;

    auto result_dequant = dequantizer.forward(test_c, packed_quantized_a.scale, packed_quantized_b.scale);
    result_dequant = result_dequant.to(torch::dtype(torch::kBFloat16));
    auto result_origin = torch::matmul(test_a, test_b.t());

    std::cout << result_dequant << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << result_origin << std::endl;

    std::cout << "Check8 Passed" << std::endl;
}

void fp32_test()
{
    torch::nn::Linear linearFP32(torch::nn::LinearOptions(2048, 4096));
    linearFP32->to(torch::kCUDA);

    torch::Tensor test_x = torch::rand({4, 2048}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);

    // CudaTimer timer;
    // timer.start();
    cudaEvent_t start_gpu, stop_gpu;

    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu, 0);
    auto forward_x_fp32 = linearFP32->forward(test_x);
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    float elapsedTime_gpu;
    cudaEventElapsedTime(&elapsedTime_gpu, start_gpu, stop_gpu);
    // LOG_INFO("FP32 Forward Time: {}ms", timer.elapsed());
    LOG_INFO("FP32 Forward Time: {}ms", elapsedTime_gpu);
}

void fp32_test2()
{
    Linear linearFP32(2048, 4096, true, torch::kFloat32);

    torch::Tensor test_x = torch::rand({4, 2048}, torch::dtype(torch::kFloat32)).to(torch::kCUDA);

    // CudaTimer timer;
    // timer.start();
    cudaEvent_t start_gpu, stop_gpu;

    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu, 0);
    auto forward_x_fp32 = linearFP32.forward(test_x);
    // timer.stop();
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    float elapsedTime_gpu;
    cudaEventElapsedTime(&elapsedTime_gpu, start_gpu, stop_gpu);
    // LOG_INFO("CutLass FP32 Forward Time: {}ms", timer.elapsed());
    LOG_INFO("CutLass FP32 Forward Time: {}ms", elapsedTime_gpu);
}

void fp16_test()
{
    Linear linearFP16(2048, 4096, true, torch::kFloat16);

    torch::Tensor test_a = torch::rand({4, 2048}, torch::dtype(torch::kFloat16)).to(torch::kCUDA);
    
    // CudaTimer timer;
    // timer.start();
    cudaEvent_t start_gpu, stop_gpu;

    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu, 0);
    auto forward_x_fp16 = linearFP16.forward(test_a);
    // timer.stop();
    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    float elapsedTime_gpu;
    cudaEventElapsedTime(&elapsedTime_gpu, start_gpu, stop_gpu);

    // LOG_INFO("FP16 Forward Time: {}ms", timer.elapsed());
    LOG_INFO("FP16 Forward Time: {}ms", elapsedTime_gpu);
}

void int4_test()
{
    Linear4bit linear4bit(2048, 4096, true);

    torch::Tensor test_a = torch::rand({4, 2048}, torch::dtype(torch::kFloat16)).to(torch::kCUDA);
    test_a *= 4;
    test_a = test_a.to(torch::dtype(torch::kBFloat16));

    Quantizer4bit quantizer(1.0);
    auto packed_quantized_a = quantizer.forward(test_a);

    CudaTimer timer;
    timer.start();
    auto forward_x = linear4bit.forward(packed_quantized_a);
    timer.stop();
    LOG_INFO("4bit Forward Time: {}ms", timer.elapsed());
}

void common_test()
{
    Linear4bit linear4bit(2048, 4096, true);
    Linear linearFP16(2048, 4096, true, torch::kFloat16);

    torch::Tensor test_a = torch::rand({4, 2048}, torch::dtype(torch::kFloat16)).to(torch::kCUDA);
    test_a *= 4;
    
    CudaTimer timer;
    timer.start();
    auto forward_x_fp16 = linearFP16.forward(test_a);
    timer.stop();
    LOG_INFO("FP16 Forward Time: {}ms", timer.elapsed());

    test_a = test_a.to(torch::dtype(torch::kBFloat16));
    Quantizer4bit quantizer(1.0);
    auto packed_quantized_a = quantizer.forward(test_a);

    // std::cout << "packed_quantized_a -> " << std::endl << packed_quantized_a.quantized_x << std::endl;
    timer.reset();
    timer.start();
    auto forward_x = linear4bit.forward(packed_quantized_a);
    timer.stop();
    // std::cout << timer.elapsed() << 'ms' << std::endl;
    LOG_INFO("4bit Forward Time: {}ms", timer.elapsed());
}

void check9()
{
    uint32_t in_dim = 8192, out_dim = 8192, batch_size = 4;
    const torch::Dtype dtype = torch::kFloat16;
    auto device = torch::kCUDA;

    torch::Tensor x = torch::rand({batch_size, in_dim}, device).to(torch::kFloat16);

    auto baseline_mod = torch::nn::Linear(torch::nn::LinearOptions(in_dim, out_dim).bias(false));
    baseline_mod->weight.data().uniform_(-8, 7).to(dtype);
    baseline_mod->to(device, dtype);
    
    auto s1_w = torch::ones({out_dim, 1}, torch::dtype(torch::kFloat16).device(device));
    auto s2_w = torch::ones({out_dim, 1}, torch::dtype(torch::kFloat16).device(device));

    auto int4_module = torch::nn::Sequential(
        Quantizer4bit(1.0),
        Linear4bitDUQ::from_float(baseline_mod, s1_w, s2_w)
    );

    auto result_1 = int4_module->forward(x);

}


int main()
{
    LoggerInit();
    LOG_INFO("{}{}Start Test ============>{}", AnsiColors::on_green, AnsiColors::white, AnsiColors::reset);
    // spdlog::info("Welcome to spdlog version !");
    // CHECK_EXEC(cudaSetDevice(3));
 
    // check1();
    // check2();
    // check3();
    // check4();
    // check5();
    // check6();
    // check7();
    // check8();
    // fp32_test();
    // fp32_test2();
    // int4_test();
    // fp16_test();
    // common_test();

    check9();

    LOG_INFO("{}{}End Test <==========={}", AnsiColors::on_green, AnsiColors::white, AnsiColors::reset);

    LoggerDrop();
    return 0;
}
