#include <iostream>
#include <vector>
#include <utility>
#include <logger.h>
#include <benchmark.h>
#include <torch/all.h>
#include <linear.h>
#include <quant_tensor.h>
#include <tabulate/table.hpp>

#include <indicators/progress_spinner.hpp>

const std::vector<std::pair<int32_t, int32_t>> model_sizes= {
    {4096, 4096},
    {5120, 5120},
    {8192, 8192}
};

const std::vector<std::pair<int32_t, int32_t>> mlp_sizes = {
    {4096, 11008},
    {5120, 13824},
    {8192, 28672}
};

const size_t warmup_steps = 5;
const size_t benchmark_steps = 1000;
const size_t batch_size = 4;
const torch::Dtype dtype = torch::kFloat16;
const std::string layer_type = "v_proj";    // current only support v_proj!

std::string to_string_with_precision(double value, int n) {
    std::string str = std::to_string(value);
    return str.substr(0, str.find(".") + n + 1);
}

template<typename T>
float module_benchmark(T &module_t, torch::Tensor &x)
{
    x = x.to(torch::kCUDA);

    LOG_INFO("{}Pre-running warmup...{}",  AnsiColors::yellow, AnsiColors::reset);

    indicators::ProgressSpinner spinner{
        indicators::option::PostfixText{"Running warmup"},
        indicators::option::ForegroundColor{indicators::Color::yellow},
        indicators::option::SpinnerStates{std::vector<std::string>{"⠈", "⠐", "⠠", "⢀", "⡀", "⠄", "⠂", "⠁"}},
        indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
    };
    for(size_t i = 0; i < warmup_steps; i++)
    {
        auto out = module_t->forward(x);
        spinner.set_progress(i * 100 / warmup_steps);
    }
    spinner.set_option(indicators::option::ForegroundColor{indicators::Color::green});
    spinner.set_option(indicators::option::PrefixText{"✔"});
    spinner.set_option(indicators::option::ShowSpinner{false});
    spinner.set_option(indicators::option::ShowPercentage{false});
    spinner.set_option(indicators::option::PostfixText{"Warmup Done!"});
    spinner.mark_as_completed();	

    torch::cuda::synchronize();

    LOG_INFO("{}Start Benchmarking...{}",  AnsiColors::red, AnsiColors::reset);

    auto start_time = std::chrono::high_resolution_clock::now();

    for(size_t i = 0; i < benchmark_steps; i++)
    {
        auto out = module_t->forward(x);
    }

    torch::cuda::synchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    LOG_INFO("{}Benchmark done, elapsed time: {} ms{}\n", AnsiColors::magenta, duration / (1000.0 * benchmark_steps), AnsiColors::reset);

    return duration / (1000.0 * benchmark_steps);
}

void linear4bit_benchmark_pipeline()
{
    echo_pipeline_head("Linear4bit Benchmark Pipeline");

    auto device = torch::kCUDA;
    // std::vector<std::pair<size_t, size_t>> &layer_size = model_sizes;
    // if (layer_type == "v_proj") layer_size = model_sizes;
    // else layer_size = mlp_sizes;

    for (auto &dims : model_sizes)
    {
        int32_t feat_dim_in = dims.first;
        int32_t feat_dim_out = dims.second;

        LOG_INFO("Current running plan: {}INPUT({}) - OUTPUT({}){}", 
                  AnsiColors::cyan, feat_dim_in, feat_dim_out, AnsiColors::reset);
        
        torch::Tensor x = torch::rand({batch_size, feat_dim_in}, device).to(torch::kFloat16);

        auto baseline_mod = torch::nn::Linear(torch::nn::LinearOptions(feat_dim_in, feat_dim_out).bias(false));
        baseline_mod->weight.data().uniform_(-8, 7).to(dtype);
        baseline_mod->to(device, dtype);
        
        auto s_w = torch::ones({feat_dim_out, 1}, torch::dtype(torch::kFloat16).device(device));

        auto int4_module = torch::nn::Sequential(
            Quantizer4bit(1.0),
            Linear4bit::from_float(baseline_mod, s_w)
        );

        std::vector<float> int4_module_time;
        std::vector<float> fp16_module_time;

        echo_test_head("Start testing Linear4bit module");

        for (int i = 0; i < 10; i++)
        {
            int4_module_time.push_back(module_benchmark(int4_module, x));
        }
        
        echo_test_head("Start testing FP16 module");
        
        for (int i = 0; i < 10; i++)
        {
            fp16_module_time.push_back(module_benchmark(baseline_mod, x));
        }
        
        tabulate::Table table;
        std::string title_str = "Test Plan: IN(" + std::to_string(feat_dim_in) + ") -> OUT(" + std::to_string(feat_dim_out) + ")";
        echo_test_head(title_str);
        tabulate::Table::Row_t header = {"Test Name"};
        for (int i = 0; i < 10; i++) header.push_back("Run #" + std::to_string(i));
        header.push_back("Average");
        table.add_row(header);

        tabulate::Table::Row_t int4_row = {"Linear4bit"};
        for (auto &time : int4_module_time) int4_row.push_back(to_string_with_precision(time, 4));
        auto int4_mean_time = to_string_with_precision(std::accumulate(int4_module_time.begin(), int4_module_time.end(), 0.0) / int4_module_time.size(), 4);
        int4_row.push_back(int4_mean_time);
        table.add_row(int4_row);

        tabulate::Table::Row_t fp16_row = {"FP16"};
        for (auto &time : fp16_module_time) fp16_row.push_back(to_string_with_precision(time, 4));
        auto fp16_mean_time = to_string_with_precision(std::accumulate(fp16_module_time.begin(), fp16_module_time.end(), 0.0) / fp16_module_time.size(), 4);
        fp16_row.push_back(fp16_mean_time);
        table.add_row(fp16_row);

        std::cout << table << std::endl;

    }
}


int main()
{
    LoggerInit();

    linear4bit_benchmark_pipeline();
    LOG_INFO("{}{}End Test <==========={}", AnsiColors::on_green, AnsiColors::white, AnsiColors::reset);
    LoggerDrop();
    return 0;
}
