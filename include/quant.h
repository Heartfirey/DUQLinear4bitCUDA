#pragma once

#include <common.h>

constexpr int QUANT_COL_K = 10;

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


void sym_double_quant_host(
    const half *x,
    const half *scale_1,
    const half *scale_2,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    Int4Storage *q
);


void sym_double_dequant_host(
    const int32_t *q,
    const half *scale_row,
    const half *scale_col_1,
    const half *scale_col_2,
    uint32_t rows,
    uint32_t cols,
    half *x
);
