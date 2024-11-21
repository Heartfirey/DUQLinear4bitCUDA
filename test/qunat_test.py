import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

n_bits = 4
n_levels = 2 ** n_bits

BATCH_SIZE = 1
TEST_N = 8
SEQ_LEN = TEST_N
FEAT_DIM_IN = 12
FEAT_DIM_OUT = TEST_N
DTYPE = torch.float16

# ^ Python simulate version of asym_quant
def asym_quant_py(x, max, min):
    delta = (max - min) / (2 ** n_bits - 1)
    zero_point = (- min / delta).round()
    # we assume weight quantization is always signed
    x_int = torch.round(x / delta)
    x_quant = torch.clamp(x_int + zero_point, 0, n_levels - 1)
    print("Py Quantized X-scale:\n", delta)
    print("Py Quantized X-zero:\n", zero_point)
    return x_quant

def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2**(bits - 1) - 1)
        minq = -maxq -1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq

def two_compl(x, bits: int):
    return torch.where(x < 0, 2 ** bits + x, x)

def pack_i4(q):
    assert torch.is_signed(q), 'The tensor to be packed should be signed int'
    minq, maxq = get_minq_maxq(4, False)
    assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)
    return q_i4

def unpack_i4(x: torch.Tensor):
    assert x.dtype == torch.uint8, 'The tensor to be unpacked should be stored in uint8'

    out_shape = list(x.shape)
    out_shape[-1] *= 2  # Each uint8 packs two numbers

    # Low 4 bits
    x0 = (x & 0x0f).to(torch.int8)
    x0[x0>=8] -= 16
    x0 = x0.view(-1, x0.shape[-1])

    # High 4 bits
    x1 = ((x & 0xf0) >> 4).to(torch.int8)
    x1[x1>=8] -= 16
    x1 = x1.view(-1, x1.shape[-1])

    out = torch.empty(out_shape, device=x.device, dtype=torch.int32)
    out = out.view(-1, out.shape[-1])
    # Interleaving
    out[:, 0::2] = x0
    out[:, 1::2] = x1

    return out.view(out_shape)
    
sample_x = torch.rand((BATCH_SIZE, SEQ_LEN, FEAT_DIM_IN)).cuda().to(dtype=DTYPE)

# print("Origin X:\n", sample_x)

py_quant_x = asym_quant_py(sample_x, torch.max(sample_x, dim=-1, keepdim=True)[0], torch.min(sample_x, dim=-1, keepdim=True)[0])

print("Py Quantized X:\n", py_quant_x)
print("Py Qunatized X-compressed:\n", pack_i4(py_quant_x.transpose(-1, -2)).transpose(-1, -2))
# print("Py Quantized X-decompressed:\n", unpack_i4(pack_i4(py_quant_x.transpose(-1, -2)).transpose(-1, -2)))

from qlinear4bit.nn.quantizer import AsymQuantizer

cuda_quant_x = AsymQuantizer()(sample_x)

print("Cuda Quantized X:\n", cuda_quant_x.quantized_x)
print("Cuda Quantized X-scales:\n", cuda_quant_x.scales_x)
print("Cuda Quantized X-zeros:\n", cuda_quant_x.zeros_x)
# print("Cuda Quantized X-decompressed:\n", unpack_i4(cuda_quant_x.quantized_x))
