from setuptools import setup
import torch.utils.cpp_extension as torch_cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os
import pathlib

os.environ['MAKEFLAGS'] = f"-j{max(int(os.cpu_count()*0.9), 1)}"
os.environ['MAX_JOBS'] = f"-j{max(int(os.cpu_count()*0.9), 1)}"

setup_dir = os.path.dirname(os.path.realpath(__file__))
HERE = pathlib.Path(__file__).absolute().parent

def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass
        
def get_cuda_arch_flags():
    return [
        # '-gencode', 'arch=compute_75,code=sm_75',  # Turing
        '-gencode', 'arch=compute_80,code=sm_80',  # Ampere
        '-gencode', 'arch=compute_86,code=sm_86',  # Ampere
    ]

def execute_cmake():
    import subprocess, sys, shutil
    cmake = shutil.which('cmake')
    if cmake is None:
        raise RuntimeError('Cannot find CMake executable.')
    os.makedirs(os.path.join(HERE, 'build'), exist_ok=True)
    retcode = subprocess.call([cmake, '--build', os.path.join(HERE, 'build')])
    if retcode != 0:
        sys.stderr.write("Error: CMake configuration failed.\n")
        sys.exit(1)
    
    # install fast hadamard transform
    hadamard_dir = os.path.join(HERE, 'lib/fast-hadamard-transform')
    pip = shutil.which('pip')
    retcode = subprocess.call([pip, 'install', '-e', hadamard_dir])

if __name__ == "__main__":
    execute_cmake()
    remove_unwanted_pytorch_nvcc_flags()
    os.makedirs(os.path.join(setup_dir, 'qlinear4bit'), exist_ok=True)
    os.makedirs(os.path.join(setup_dir, 'packages', 'qlinear4bit'), exist_ok=True)
    setup(
        name='packages.qlinear4bit',
        ext_modules=[
            CUDAExtension(
                name='packages.qlinear4bit._CUDA',
                sources=[
                    'src/export/bindings.cpp',
                    'src/kernels/gemm.cu',
                    'src/kernels/quant.cu',
                    'src/kernels/quant2.cu',
                    'src/kernels/flashinfer.cu',
                ],
                libraries=['cudart'],
                include_dirs=[
                    os.path.join(setup_dir, 'include'),
                    os.path.join(setup_dir, 'lib/cutlass/include'),
                    os.path.join(setup_dir, 'lib/cutlass/tools/util/include'),
                ],
                extra_compile_args={
                    'cxx': ["-w"],
                    'nvcc': get_cuda_arch_flags()+["--threads", "4", "-w"],
                }
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
