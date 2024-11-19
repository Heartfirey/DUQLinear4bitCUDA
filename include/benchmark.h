#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <tabulate/table.hpp>

class CudaTimer
{
public:
    cudaEvent_t startEvent, stopEvent;
    CudaTimer();
    ~CudaTimer();

    void start();
    void stop();
    void reset();
    float elapsed();
};

class ResultTable
{
public:
    ResultTable(const std::vector<std::string> task);
    void addData(const std::vector<std::string> value);
    void generateTable();
private:
    tabulate::Table table;
    int tasks=0;
};

void echo_pipeline_head(const std::string pipeline_name);

void echo_test_head(const std::string test_name);
