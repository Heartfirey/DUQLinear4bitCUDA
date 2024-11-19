import math
import torch
# import fast_hadamard_transform
import python.tools.tensor_utils as tensor_utils
import python.tools.quantization as quant_utils

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


class Linear4bit(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        '''
        Symmetric 4-bit Linear Layer.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scales',
                             torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', (torch.randint(1, 7, (self.out_features, self.in_features // 2),
                                                             # SubByte weight
                                                             dtype=torch.uint8, requires_grad=False)))
        if bias:                                                        
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None
        
    def forward(self, x):
        #if torch.cuda.current_device() != x.device:
        #    torch.cuda.set_device(x.device)
        
        assert type(x) == tensor_utils.PackedQuantizedTensor #Quantized input is given
        x, scales_x = x.quantized_x, x.scales_x
        #shape_handler = ShapeHandler(quantized_x)
        #quantized_x = shape_handler.flatten(quantized_x)
        x = tensor_utils.matmul(x, self.weight)
        #out = shape_handler.unflatten(
        #    quarot.sym_dequant(int_result, scales_x, self.weight_scales))
        if self.bias is not None:
            return tensor_utils.sym_dequant(x, scales_x, self.weight_scales) + self.bias
        else:
            return tensor_utils.sym_dequant(x, scales_x, self.weight_scales)

    @staticmethod
    def from_float(module: torch.nn.Linear, weight_scales=None,):
        '''
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        weight_matrix = module.weight.data
        
        int_module = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, dtype=weight_matrix.dtype).to(weight_matrix.dtype)
        if weight_scales is not None:
            assert weight_scales.shape == (module.out_features, 1), 'weight_scales should have shape (out_features, 1)'
            weight_matrix = weight_matrix.cuda()
            int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
            int_rounded_weight = (weight_matrix/weight_scales.cuda()).round()
            int_module.weight.copy_(quant_utils.pack_i4(int_rounded_weight.to(torch.int8)).cpu())
        
            if module.bias is not None:
                int_module.bias.copy_(module.bias)
        
        return int_module

class Linear4bitDUQ(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        '''
        Symmetric 4-bit Linear Layer.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scales_1',
                             torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight_scales_2',
                             torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', (torch.randint(1, 7, (self.out_features, self.in_features // 2),
                                                             # SubByte weight
                                                             dtype=torch.uint8, requires_grad=False)))
        if bias:                                                        
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None

    def forward(self, x):
        
        assert type(x) == tensor_utils.PackedQuantizedTensor
        x, scales_x = x.quantized_x, x.scales_x
        x = tensor_utils.matmul(x, self.weight)
        
        if self.bias is not None:
            return tensor_utils.sym_dual_dequant(x, scales_x, self.weight_scales_1, self.weight_scales_2) + self.bias
        else:
            return tensor_utils.sym_dual_dequant(x, scales_x, self.weight_scales_1, self.weight_scales_2)

    @staticmethod
    def from_float(module: torch.nn.Linear, weight_scales_1=None, weight_scales_2=None):
        '''
        Generate a new Linear4bit module from a FP16 Linear module. (Quantized using Dual  Quantization)
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        weight_matrix = module.weight.data
        
        int_module = Linear4bitDUQ(module.in_features, module.out_features, bias=module.bias is not None, dtype=weight_matrix.dtype).to(weight_matrix.dtype)
        if weight_scales_1 is not None and weight_scales_2 is not None:
            assert weight_scales_1.shape == (module.out_features, 1), 'weight_scales_1 should have shape (out_features, 1)'
            assert weight_scales_2.shape == (module.out_features, 1), 'weight_scales_2 should have shape (out_features, 1)'
            weight_matrix = weight_matrix.cuda()
            int_module.weight_scales_1.copy_(weight_scales_1.to(weight_matrix.dtype))
            int_module.weight_scales_2.copy_(weight_scales_2.to(weight_matrix.dtype))
            int_rounded_weight = (weight_matrix/weight_scales_1.cuda()).round()
            int_module.weight.copy_(quant_utils.pack_i4(int_rounded_weight.to(torch.int8)).cpu())
        
            if module.bias is not None:
                int_module.bias.copy_(module.bias)
        
        return int_module
        
