# DUQLinear4bitCUDA

> This project partially based on the code from the [Quarot](https://github.com/spcl/QuaRot) project.

CUDA Implmentation of Linear4bit layer with Dual-Uniform-Quantization support.

# 1. Setup

This project provides C++ API and Python API. for Python API, you can refer to the following way to install it:

```
mkdir build && cd build
cmake .. && make -j <NUM_PROCESS> && cd ..
pip install -e . --verbose
```

Then the cuda depedencies will be compile into the `qlinear4bit`:

```
_CUDA.cpython-39-xxxxx.so
```

We provided a pre-compile version of  `cpython-39-x86_64-linux` in the release page, you can download it into the `qlinear4bit` folder directly.

## 2. Environment

- `python 3.9`
- `cuda 12.1`
- `torch2.5.1-cu121`
