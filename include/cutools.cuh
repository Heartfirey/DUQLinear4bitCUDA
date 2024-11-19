#pragma once
#include <stdio.h>
#include <common.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CHECK_EXEC(call)                              \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)


#define CHECK_CUTLASS(call)                           \
do {                                                  \
    auto status = call;                               \
    ensure(status == cutlass::Status::kSuccess,       \
           cutlassGetStatusString(status));           \
} while (0)

// quick create a tensor
template <typename T>
T *NewTensor(uint32_t *shape, uint32_t dimensions)
{
    uint32_t totElements = 1;
    for (uint32_t i = 0; i < dimensions; i++)
    {
        totElements *= shape[i];
    }
    // calculate the tensor size in bytes
    uint32_t sizeInBytes = totElements * sizeof(T);
    // call cudaMalloc
    T *dTensor;
    CHECK_EXEC(cudaMalloc((void **)&dTensor, sizeInBytes));

    return dTensor;
}

// quick create a tensor and assign values from host ptr
template <typename T>
T *NewTensorFromHost(const T *hostData, uint32_t *shape, uint32_t dimensions)
{
    uint32_t totElements = 1;
    for (uint32_t i = 0; i < dimensions; i++)
    {
        totElements *= shape[i];
    }
    // calculate the tensor size in bytes
    uint32_t sizeInBytes = totElements * sizeof(T);
    // call cudaMalloc
    T *dTensor;
    CHECK_EXEC(cudaMalloc((void **)&dTensor, sizeInBytes));
    // call cudaMemcpy
    CHECK_EXEC(cudaMemcpy(dTensor, hostData, sizeInBytes, cudaMemcpyHostToDevice));
    
    return dTensor;
}

template <typename T>
half *NewFP16TensorFromHost(const T *hostData, uint32_t *shape, uint32_t dimensions)
{
    uint32_t totElements = 1;
    for (uint32_t i = 0; i < dimensions; i++)
    {
        totElements *= shape[i];
    }

    half *dTensor;
    CHECK_EXEC(cudaMalloc((void **)&dTensor, totElements * sizeof(half)));
}

