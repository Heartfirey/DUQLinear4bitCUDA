#pragma once

#include <common.h>

constexpr int QUANT_COL_K = 10;

constexpr __uint128_t QUANT_COL_ST = ((__int128)1 << 0) | ((__int128)1 << 1) | ((__int128)1 << 2) | ((__int128)1 << 3) | ((__int128)1 << 4) | ((__int128)1 << 10) | ((__int128)1 << 11) | ((__int128)1 << 12) | ((__int128)1 << 13) | ((__int128)1 << 14);

void sym_quant_host(
        const half *x,
        const half *scale,
        uint32_t rows,
        uint32_t colsSrc,
        uint32_t colsDst,
        Int4Storage *q
);


void sym_dequant_host(
        const int32_t *q,
        const half *scale_row,
        const half *scale_col,
        uint32_t rows,
        uint32_t cols,
        half *x
);


void sym_dual_quant_host(
    const half *x,
    const half *scale_1,
    const half *scale_2,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    Int4Storage *q
);


void sym_dual_dequant_host(
    const int32_t *q,
    const half *scale_row,
    const half *scale_col_1,
    const half *scale_col_2,
    uint32_t rows,
    uint32_t cols,
    half *x
);

void asym_quant_host(
    const half *x,
    const half *scale,
    const half *zeros,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    Int4Storage *q
);

void asym_dequant_host(
    const int32_t *q,
    const half *scale_row,
    const half *zeros_row,
    const half *scale_col,
    const half *zeros_col,
    uint32_t rows,
    uint32_t cols,
    half *x
);

void asym_quant_host_8bit(
    const float *x,
    const float *scale,
    const float *zeros,
    uint32_t rows,
    uint32_t cols,
    int8_t *q 
);

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
);

void asym_dual_quant_host(
    const half *x,
    const half *scale_1,
    const half *scale_2,
    const half *zeros_1,
    const half *zeros_2,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    Int4Storage *q
);


void asym_dual_dequant_host(
    const int32_t *q,
    const half *scale_row,
    const half *zeros_row,
    const half *scale_col_1,
    const half *zeros_col_1,
    const half *scale_col_2,
    const half *zeros_col_2,
    uint32_t rows,
    uint32_t cols,
    half *x
);

void asym_dequant_host_hprec(
    const int32_t *q,
    const float *scale_row,
    const float *zeros_row,
    const float *scale_col,
    const float *zeros_col,
    uint32_t rows,
    uint32_t cols,
    float *x
);
