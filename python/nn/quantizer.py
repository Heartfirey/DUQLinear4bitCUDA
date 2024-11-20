import torch
# import python.tools.quantization as quant_utils
import python.tools.tensor_utils as tensor_utils

class SymQuantizer(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio
    
    def forward(self, x) -> tensor_utils.PackedQuantizedTensor:
        scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1)/7).to(torch.float16) * self.input_clip_ratio
        quantized_x = tensor_utils.sym_quant(x, scales_x)
        packed_tensor = tensor_utils.PackedQuantizedTensor(quantized_x, scales_x)
        return packed_tensor

class AsymQuantizer(torch.nn.Module):
    def __init__(self, n_bits: int=4, input_clip_ratio: float=1.0):
        super().__init__()
        self.n_levels = 2**n_bits
        self.input_clip_ratio = input_clip_ratio

    def forward(self, x) -> tensor_utils.PackedQuantizedTensor:
        min_val, _ = x.min(dim=-1, keepdim=True)
        max_val, _ = x.max(dim=-1, keepdim=True)
        
        scale = (max_val - min_val) / (self.n_levels - 1)
        zeros = (-min_val / scale).round().to(torch.float16)
        
        quantized_x = tensor_utils.asym_quant(x, scale, zeros)
        packed_tensor = tensor_utils.PackedQuantizedTensor(quantized_x, scale, zeros)
        return packed_tensor
