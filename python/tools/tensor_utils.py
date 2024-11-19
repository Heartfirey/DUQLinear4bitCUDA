import torch
import packages.qlinear4bit._CUDA as qcu_tool

class ShapeHandler:
    def __init__(self, x: torch.Tensor):
        self.size_excl_last = x.numel()//x.shape[-1]
        self.shape_excl_last = tuple(x.shape[:-1])

    # Keep the last dim unchanged, flatten all previous dims
    def flatten(self, x: torch.Tensor):
        return x.view(self.size_excl_last, -1)

    # Recover back to the original shape.
    def unflatten(self, x: torch.Tensor):
        return x.view(self.shape_excl_last + (-1,))

    def unflatten_scale(self, x: torch.Tensor):
        return x.view(self.shape_excl_last)

def flatten_last_dim_and_return_shape(x: torch.Tensor):
    shape_excl_last = x.shape[:-1]
    x = x.view(-1, x.shape[-1])
    return x, shape_excl_last


def matmul(A, B):
    assert A.shape[-1] % 32 == 0, "A.shape[-1]: {} must be multiplication of 32".format(A.shape[-1])
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    B, B_shape_excl_last = flatten_last_dim_and_return_shape(B)
    return qcu_tool.matmul(A, B).view(*A_shape_excl_last, *B_shape_excl_last)

def sym_quant(x, scale):
    assert x.dtype == scale.dtype == torch.float16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return qcu_tool.sym_quant(x, scale.view(-1)).view(*x_shape_excl_last, -1)

def sym_dual_quant(x, scale_1, scale_2):
    assert x.dtype == scale_1.dtype == scale_2.dtype == torch.float16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return qcu_tool.sym_double_quant(x, scale_1.view(-1), scale_2.view(-1)).view(*x_shape_excl_last, -1)

def sym_dequant(q, scale_row, scale_col, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == scale_col.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return qcu_tool.sym_dequant(q, scale_row.view(-1), scale_col, bits).view(*q_shape_excl_last, -1)

def sym_dual_dequant(q, scale_row, scale_col_1, scale_col_2, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == scale_col_1.dtype == scale_col_2.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return qcu_tool.sym_double_dequant(q, scale_row.view(-1), scale_col_1, scale_col_2, bits).view(*q_shape_excl_last, -1)

class PackedQuantizedTensor:
    def __init__(self, 
                 quantized_x: torch.Tensor, 
                 scales_x: torch.Tensor):
        self.quantized_x = quantized_x
        self.scales_x = scales_x

    def size(self):
        return self.quantized_x.size()
    
    @property
    def device(self):
        return self.quantized_x.device
    
    @property
    def dtype(self):
        return self.quantized_x.dtype
