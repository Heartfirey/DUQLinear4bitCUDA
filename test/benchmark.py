import torch
from python.nn import Linear4bit, Linear4bitDUSQ, SymQuantizer
import time
import argparse
import numpy as np
import pprint

model_sizes = [
    (4096, 4096), #llama-7b
    (5120, 5120), #llama-13b
    (8192, 8192)  #llama-70b   
]

mlp_sizes = [
    (4096, 11008), #llama-7b
    (5120, 13824), #llama-13b
    (8192, 28672)  #llama-70b
]
benchmark_dtypes = [torch.float16]
num_warmup_steps = 5
num_bench_steps = 100


def module_benchmark(module, x):
    x = x.cuda()
    
    # warmup
    for i in range(num_warmup_steps):
        out = module(x)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        out = module(x)
    torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000 / num_bench_steps

def linear4bit_benchmark(args):
    bsz = args.bsz
    seq_len = args.seq_len
    
    if args.layer_type == 'v_proj':
        layer_size = model_sizes
    else:
        layer_size = mlp_sizes
        
    
    for (feature_dim_in, feature_dim_out) in layer_size:
        for dtype in benchmark_dtypes:
            
            x = torch.rand((bsz,
                            seq_len,
                            feature_dim_in)).cuda().to(dtype)
            
            baseline_mod = torch.nn.Linear(feature_dim_in,
                                           feature_dim_out,
                                           bias=False).cuda().to(dtype)
            
            baseline_mod.weight.data = torch.randint_like(baseline_mod.weight.data,
                                                          low=-8, high=7).to(dtype)
            
            s_w = torch.ones((feature_dim_out, 1), dtype=torch.float16, device='cuda')
            s_w2 = torch.ones((feature_dim_out, 1), dtype=torch.float16, device='cuda')
            int4_mod = torch.nn.Sequential(
                SymQuantizer(input_clip_ratio=1.0),
                Linear4bit.from_float(baseline_mod, weight_scales=s_w)
            ).cuda()
            int4duq_mod = torch.nn.Sequential(
                SymQuantizer(input_clip_ratio=1.0),
                Linear4bitDUSQ.from_float(baseline_mod, weight_scales_1=s_w, weight_scales_2=s_w2)
            ).cuda()


            print(f"{dtype}. Sizes: {baseline_mod.weight.shape}")
            times_4bit = []
            for i in range(10):
                times_4bit.append(module_benchmark(int4_mod, x))
            print(f"Int4 time: {np.mean(times_4bit):.3f} +- {1.96 * np.std(times_4bit):.3f}ms")
            
            times_4bitduq = []
            for i in range(10):
                times_4bitduq.append(module_benchmark(int4duq_mod, x))
            print(f"Int4-DualUniformQuant time: {np.mean(times_4bitduq):.3f} +- {1.96 * np.std(times_4bitduq):.3f}ms")
            

            
            times_baseline = []
            for i in range(10):
                times_baseline.append(module_benchmark(baseline_mod, x))
            print(f"FP16 time: {np.mean(times_baseline):.3f} +- {1.96 * np.std(times_baseline):.3f}ms")
            
            print(f"Speedup(baeline -> Int4): {np.mean(times_baseline) / np.mean(times_4bit):.3f}x")
            print(f"Speedup(baeline -> Int4-DUQ): {np.mean(times_baseline) / np.mean(times_4bitduq):.3f}x")
            
            # table-style output
            print(f'{feature_dim_in}x{feature_dim_out} & {args.bsz} & {np.mean(times_baseline):.3f} & {np.mean(times_4bit):.3f}\\\\')
            print('--------------')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bsz', type=int,
        help='Batch size',
        default=1,
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    parser.add_argument(
        '--layer_type', type=str,
        help='Type of the layer in the model (v_proj [default], down_proj)',
        default='v_proj',
        choices=['v_proj', 'down_proj']
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    linear4bit_benchmark(args)
