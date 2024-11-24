
import math
import torch
import torch.nn as nn

from qlinear4bit.nn.quantizer import SymQuantizer, AsymQuantizer
import qlinear4bit.tools.tensor_utils as tensor_utils

def pad2multiple(x: torch.Tensor, pad_dim: int=-1, multiple: int=32):
    cur_size = x.size()[pad_dim]
    pad_need = (multiple - cur_size % multiple) % multiple
    pad_config = [0, 0] * (len(x.size()) - pad_dim - 1) + [0, pad_need]
    return torch.nn.functional.pad(x, pad_config), pad_need

def unpad(x: torch.Tensor, pad_dim, pad_need):
    size = list(x.size())
    size[pad_dim] -= pad_need
    return x.narrow(pad_dim, 0, size[pad_dim]).contiguous()

class AsymQuantMatMul(nn.Module):
    def __init__(self, transpose_B=False):
        super(AsymQuantMatMul, self).__init__()
        self.quantizer_A = AsymQuantizer()
        self.quantizer_B = AsymQuantizer()
        self.transpose_B = transpose_B
        
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if self.transpose_B:
            B = B.transpose(-1, -2)
        EXT_DIM_A, HA, WA = A.shape[:-2], A.shape[-2], A.shape[-1]
        EXT_DIM_B, HB, WB = B.shape[:-2], B.shape[-2], B.shape[-1]
        for EXT_DIM_Ai, EXT_DIM_Bi in zip(EXT_DIM_A, EXT_DIM_B):
            assert EXT_DIM_Ai == EXT_DIM_Bi, "Batch dimensions must match"
        
        EXT_DIM_SUM = math.prod(EXT_DIM_A)
        
        A = A.reshape(EXT_DIM_SUM, HA, WA)
        B = B.reshape(EXT_DIM_SUM, HB, WB)
        
        A = self.quantizer_A(A)
        B = self.quantizer_B(B)
        
        WA = WB = WA//2
        
        A, scales_A, zeros_A = A.quantized_x, A.scales_x, A.zeros_x
        B, scales_B, zeros_B = B.quantized_x, B.scales_x, B.zeros_x
        
        PAD_HA, PAD_HB = 0, 0
        # if HA % 32 != 0:
        #     A, pad_need = pad2multiple(A, pad_dim=1, multiple=32)
        #     PAD_HA = pad_need
        if HB % 32 != 0:
            B, pad_need = pad2multiple(B, pad_dim=1, multiple=32)
            PAD_HB = pad_need
        
        C = torch.zeros((EXT_DIM_SUM, HA, HB + PAD_HB), device=A.device, dtype=torch.int32)
        
        for bs_idx in range(EXT_DIM_SUM):
            cur_A, cur_B = A[bs_idx], B[bs_idx]
            cur_prod = tensor_utils.matmul(cur_A, cur_B)
            # if PAD_HB != 0: cur_prod = unpad(cur_prod, -1, PAD_HB).contiguous()
            C[bs_idx].copy_(cur_prod)
        
        if PAD_HB != 0:
            C = unpad(C, -1, PAD_HB)
        C = tensor_utils.asym_batch_dequant(C, scales_A, zeros_A, scales_B, zeros_B)
        
        return C.view(list(EXT_DIM_A) + [HA, HB])
        


class SymQuantMatMul(nn.Module):
    def __init__(self):
        super(SymQuantMatMul, self).__init__()
        self.quantizer_A = SymQuantizer()
        self.quantizer_B = SymQuantizer()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = self.quantizer_A(A)
        B = self.quantizer_B(B)
        
        out = tensor_utils.matmul(A.quantized_x, B.quantized_x)
        out = tensor_utils.sym_dequant(out, A.scales_x, B.scales_x)
        return out
