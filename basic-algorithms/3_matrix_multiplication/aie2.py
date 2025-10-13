# section-3/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1

if len(sys.argv) > 1:
    if sys.argv[1] == "npu":
        dev = NPU1Col1()
    elif sys.argv[1] == "npu2":
        dev = NPU2Col1()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))
    
M = 3
K = 3
N = 3

tensor_M_ty = np.ndarray[(M * K,), np.dtype[np.int32]]
tensor_N_ty = np.ndarray[(K * N,), np.dtype[np.int32]]
tensor_RES_ty = np.ndarray[(M * N,), np.dtype[np.int32]]

# External, binary kernel definition
mat_mul_fn = Kernel(
    "matrix_mul_aie",
    "mat_mul.o",
    [tensor_M_ty, tensor_N_ty, tensor_RES_ty, np.int32, np.int32, np.int32],
)

# Input data movement
of_in_M = ObjectFifo(tensor_M_ty, name="in_M")
of_in_N = ObjectFifo(tensor_N_ty, name="in_N")

# Output data movement
of_out = ObjectFifo(tensor_RES_ty, name="out")

# Task for the core to perform
def core_fn(of_in_M, of_in_N, of_out, mat_mul_fn):
    elem_in_M = of_in_M.acquire(1)
    elem_in_N = of_in_N.acquire(1)
    elem_out = of_out.acquire(1)
    
    mat_mul_fn(elem_in_M, elem_in_N, elem_out, M, K, N)
    
    of_in_M.release(1)
    of_in_N.release(1)
    of_out.release(1)


# Create a worker to perform the task
my_worker = Worker(core_fn, [of_in_M.cons(), of_in_N.cons(), of_out.prod(), mat_mul_fn])

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(tensor_M_ty, tensor_N_ty, tensor_RES_ty) as (M_in, N_in, RES_out):
    rt.start(my_worker)
    rt.fill(of_in_M.prod(), M_in)
    rt.fill(of_in_N.prod(), N_in)
    rt.drain(of_out.cons(), RES_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
