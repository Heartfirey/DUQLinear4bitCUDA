import torch
from qlinear4bit.nn import Linear4bit, Linear4bitDUSQ, LinearQuant4bitDUASQ, SymQuantizer, AsymQuantizer
import qlinear4bit.tools.tensor_utils as tensor_utils

BATCH_SIZE = 1
SEQ_LEN = 2048
FEAT_DIM_IN = 2048
FEAT_DIM_OUT = 2048
DTYPE = torch.float16

x = torch.rand((BATCH_SIZE, SEQ_LEN, FEAT_DIM_IN)).cuda().to(dtype=DTYPE)
# baseline_mod = torch.nn.Linear(FEAT_DIM_IN, FEAT_DIM_OUT, bias=False).cuda().to(dtype=DTYPE)
# baseline_mod.weight.data = torch.randint_like(baseline_mod.weight.data, low=-8, high=7).to(dtype=DTYPE)

# s_w = torch.ones((FEAT_DIM_IN, 1), dtype=DTYPE, device='cuda')

# int4_mod = torch.nn.Sequential(
#     SymQuantizer(input_clip_ratio=1.0),
#     Linear4bit.from_float(baseline_mod, weight_scales=s_w)
# ).cuda()

# int4d_mod = torch.nn.Sequential(
#     SymQuantizer(input_clip_ratio=1.0),
#     Linear4bitDUSQ.from_float(baseline_mod, weight_scales_1=s_w, weight_scales_2=s_w)
# ).cuda()

# out1 = baseline_mod(x)

# # test_quantizer = Quantizer(input_clip_ratio=1.0)
# # quant_x = test_quantizer(x)
# # print(torch.max(quant_x.quantized_x))
# # print(torch.min(quant_x.quantized_x))

# out2 = int4_mod(x)

# out3 = int4d_mod(x)

# print(out1)
# print(out2)
# print(out3)

# torch.cuda.synchronize()
# print("Test1 done.")

# int4da_mod = LinearQuant4bitDUASQ.from_linear(baseline_mod).cuda()
# # import IPython; IPython.embed(); 
# out4 = int4da_mod(x)
# print(out4)

# torch.cuda.synchronize()
# print("Test2 done.")


test_asym_quantizer_1 = AsymQuantizer(input_clip_ratio=1.0)
test_asym_quantizer_2 = AsymQuantizer(input_clip_ratio=1.0)

x1 = test_asym_quantizer_1(x)
x2 = test_asym_quantizer_2(x)

x_mul = tensor_utils.matmul(x1.quantized_x, x2.quantized_x)

print(x_mul)

# import IPython; IPython.embed()

x_mul_dequant = tensor_utils.asym_dequant(x_mul, x1.scales_x, x1.zeros_x, x2.scales_x, x2.zeros_x)

print(x_mul_dequant)
