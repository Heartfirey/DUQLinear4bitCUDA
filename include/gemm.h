#pragma once

#include <common.h>

void matmul_host_4bit(
        const Int4Storage *A,
        const Int4Storage *B,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        int32_t *C
);

void batch_matmul_host_4bit(
    const Int4Storage **A,
    const Int4Storage **B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int32_t **C,
    uint32_t batch_count
);

template <typename T>
void matmul_host(
    const T *A,
    const T *B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    T *C
);

template <typename T>
__global__ void add_bias(
    T *C,
    T *bias,
    uint32_t batch_size,
    uint32_t feat_dim
);
