import torch
import torch.nn as nn

from qlinear4bit.nn.quantizer import SymQuantizer, AsymQuantizer
import qlinear4bit.tools.tensor_utils as tensor_utils


class AsymQuantMatMul(nn.Module):
    def __init__(self):
        super(AsymQuantMatMul, self).__init__()
        self.quantizer_A = AsymQuantizer()
        self.quantizer_B = AsymQuantizer()
        

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = self.quantizer_A(A)
        B = self.quantizer_B(B)
        
        out = tensor_utils.matmul(A.quantized_x, B.quantized_x)
        out = tensor_utils.asym_dequant(out, A.scales_x, A.zeros_x, B.scales_x, B.zeros_x)
        return out


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
