#include <benchmark.h>

CudaTimer::CudaTimer()
{
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
}

CudaTimer::~CudaTimer()
{
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void CudaTimer::start()
{
    cudaEventRecord(startEvent, 0);
}

void CudaTimer::stop()
{
    cudaEventRecord(stopEvent, 0);
}

float CudaTimer::elapsed()
{
    float milliseconds = 0;
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    return milliseconds;
}

void CudaTimer::reset()
{
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
}

ResultTable::ResultTable(std::vector<std::string> task)
{
    int n = task.size();
    this->tasks = n;
    for (size_t i = 0; i < n; i++) 
    {
        table.column(i).format().font_align(tabulate::FontAlign::center);
    }
    tabulate::Table::Row_t task_r;
    for(auto x : task) task_r.push_back(x);
    table.add_row(task_r);
}

void ResultTable::addData(std::vector<std::string> value)
{
    tabulate::Table::Row_t value_r;
    for(auto x : value) value_r.push_back(x);
    table.add_row(value_r);
}

void ResultTable::generateTable()
{
    int n = this->tasks;
    for (size_t i = 0; i < n; i++)
    {
        table[0][i].format()
            .font_color(tabulate::Color::yellow)
            .font_align(tabulate::FontAlign::center)
            .font_style({tabulate::FontStyle::bold});
    }
}

void echo_pipeline_head(const std::string pipeline_name)
{
    tabulate::Table main_t;
    main_t.format()
            .border_color(tabulate::Color::green)
            .font_color(tabulate::Color::white)
            .font_align(tabulate::FontAlign::center)
            .width(60);
    main_t.add_row(tabulate::Table::Row_t{"Efficient Linear 4bit C++ Benchmark"});
    main_t[0].format().font_background_color(tabulate::Color::green);
    main_t.add_row(tabulate::Table::Row_t{pipeline_name});
    main_t[1].format().font_style({tabulate::FontStyle::underline});
    std::cout << main_t << std::endl;
}

void echo_test_head(const std::string test_name)
{
    tabulate::Table test_t;
    test_t.format()
            .border_color(tabulate::Color::green)
            .font_color(tabulate::Color::white)
            .font_align(tabulate::FontAlign::center)
            .width(60);
    test_t.add_row(tabulate::Table::Row_t{test_name});
    test_t[0].format().font_background_color(tabulate::Color::cyan);
    std::cout << test_t << std::endl;
}
