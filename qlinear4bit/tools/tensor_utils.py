import torch
import qlinear4bit._CUDA as qcu_tool

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
    
class PackedQuantizedTensor:
    def __init__(self, 
                 quantized_x: torch.Tensor, 
                 scales_x: torch.Tensor,
                 zeros_x: torch.Tensor = None):
        self.quantized_x = quantized_x
        self.scales_x = scales_x
        self.zeros_x = zeros_x

    def size(self):
        return self.quantized_x.size()
    
    @property
    def device(self):
        return self.quantized_x.device
    
    @property
    def dtype(self):
        return self.quantized_x.dtype
    
    
def flatten_last_dim_and_return_shape(x: torch.Tensor):
    shape_excl_last = x.shape[:-1]
    x = x.view(-1, x.shape[-1])
    return x, shape_excl_last

def flatten_last_two_dim_and_return_shape(x: torch.Tensor):
    shape_excl_last_two, shape_last_two = x.shape[:-2], x.shape[-2:]
    x = x.view(-1, x.shape[-2], x.shape[-1])
    return x, shape_excl_last_two, shape_last_two

def matmul(A, B):
    assert A.shape[-1] % 32 == 0, "A.shape[-1]: {} must be multiplication of 32".format(A.shape[-1])
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    B, B_shape_excl_last = flatten_last_dim_and_return_shape(B)
    return qcu_tool.matmul(A, B).view(*A_shape_excl_last, *B_shape_excl_last)

def matmul_8bit(A, B):
    assert A.shape[-1] % 32 == 0, "A.shape[-1]: {} must be multiplication of 32".format(A.shape[-1])
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    B, B_shape_excl_last = flatten_last_dim_and_return_shape(B)
    return qcu_tool.matmul_8bit(A, B).view(*A_shape_excl_last, *B_shape_excl_last)

def batched_matmul(A, B):
    assert A.shape[-1] % 32 == 0, "A.shape[-1]: {} must be multiplication of 32".format(A.shape[-1])
    # A, A_shape_excl_last_two, A_shape_last_two = flatten_last_two_dim_and_return_shape(A)
    # B, B_shape_excl_last_two, B_shape_last_two = flatten_last_two_dim_and_return_shape(B)
    import IPython; IPython.embed();
    return qcu_tool.batched_matmul(A, B)

def sym_quant(x, scale):
    assert x.dtype == scale.dtype == torch.float16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return qcu_tool.sym_quant(x, scale.view(-1)).view(*x_shape_excl_last, -1)

def sym_dual_quant(x, scale_1, scale_2):
    assert x.dtype == scale_1.dtype == scale_2.dtype == torch.float16
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    return qcu_tool.sym_dual_quant(x, scale_1.view(-1), scale_2.view(-1)).view(*x_shape_excl_last, -1)

def sym_dequant(q, scale_row, scale_col, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == scale_col.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return qcu_tool.sym_dequant(q, scale_row.view(-1), scale_col, bits).view(*q_shape_excl_last, -1)

def sym_dual_dequant(q, scale_row, scale_col_1, scale_col_2, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == scale_col_1.dtype == scale_col_2.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return qcu_tool.sym_dual_dequant(q, scale_row.view(-1), scale_col_1, scale_col_2, bits).view(*q_shape_excl_last, -1)

def asym_quant(q, scale, zero):
    assert q.dtype == torch.float16
    assert scale.dtype == zero.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return qcu_tool.asym_quant(q, scale.view(-1), zero.view(-1)).view(*q_shape_excl_last, -1)

def asym_quant_8bit(q, scale, zero):
    assert q.dtype == torch.float16
    assert scale.dtype == zero.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return qcu_tool.asym_quant_8bit(q, scale.view(-1), zero.view(-1)).view(*q_shape_excl_last, -1)

def asym_dequant(q, scale_row, zeros_row, scale_col, zeros_col, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == zeros_row.dtype == scale_col.dtype == zeros_col.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return qcu_tool.asym_dequant(q, scale_row.view(-1), zeros_row.view(-1), scale_col.view(-1), zeros_col.view(-1), bits).view(*q_shape_excl_last, -1)

def asym_dequant_hprec(q, scale_row, zeros_row, scale_col, zeros_col, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == zeros_row.dtype == scale_col.dtype == zeros_col.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return qcu_tool.asym_dequant_hprec(q, scale_row.view(-1), zeros_row.view(-1), scale_col.view(-1), zeros_col.view(-1), bits).view(*q_shape_excl_last, -1)

def asym_batch_dequant(q, scale_row, zeros_row, scale_col, zeros_col, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == zeros_row.dtype == scale_col.dtype == zeros_col.dtype == torch.float16
    q, q_shape_excl_last_two, q_shape_last_two = flatten_last_two_dim_and_return_shape(q)
    return qcu_tool.asym_batch_dequant(q, 
                                       scale_row, zeros_row, 
                                       scale_col, zeros_col, bits
                                       ).view(*q_shape_excl_last_two, q_shape_last_two[-2], q_shape_last_two[-1])

def asym_dual_quant(q, scale_1, zeros1, scale_2, zeros2):
    assert q.dtype == torch.int32
    assert scale_1.dtype == scale_2.dtype == torch.float16
    assert zeros1.dtype == zeros2.dtype == torch.float16
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return qcu_tool.asym_dual_quant(q, 
                                    scale_1.view(-1), zeros1.view(-1), 
                                    scale_2.view(-1), zeros2.view(-1)).view(*q_shape_excl_last, -1)

def asym_dual_dequant(q, scale_row, zeros_row, scale_col_1, zeros_col_1, scale_col_2, zeros_col_2, bits=32):
    assert q.dtype == torch.int32
    assert scale_row.dtype == scale_col_1.dtype == scale_col_2.dtype == torch.float16, "torch.float16 assertation error: scale_row.dtype: {}, scale_col_1.dtype: {}, scale_col_2.dtype: {}".format(scale_row.dtype, scale_col_1.dtype, scale_col_2.dtype)
    assert zeros_row.dtype == zeros_col_1.dtype == zeros_col_2.dtype == torch.float16, "torch.float16 assertation error: zeros_row.dtype: {}, zeros_col_1.dtype: {}, zeros_col_2.dtype: {}".format(zeros_row.dtype, zeros_col_1.dtype, zeros_col_2.dtype)
    q, q_shape_excl_last = flatten_last_dim_and_return_shape(q)
    return qcu_tool.asym_dual_dequant(q, 
                                      scale_row.view(-1), zeros_row.view(-1), 
                                      scale_col_1.view(-1), zeros_col_1.view(-1), 
                                      scale_col_2.view(-1), zeros_col_2.view(-1), bits).view(*q_shape_excl_last, -1)

def get_dual_quant_col():
    return qcu_tool.get_dual_quant_col_k()
