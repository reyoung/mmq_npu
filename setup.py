import os
import sys

import torch
from skbuild import setup

torch_root = os.path.dirname(torch.__file__)

setup(
    name="mmq_npu",
    version="0.0.1",
    description="NPU ops for mmq",
    license="MIT",
    packages=["mmq_npu"],
    cmake_args=[
        f"-DCMAKE_PREFIX_PATH={torch_root}"
    ],  # to specify CUDA location: -DCMAKE_CUDA_COMPILER=...
)
