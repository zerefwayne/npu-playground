"""
•	Kernels:
    •	A: Producer generates two input vectors (x and y) and sends them via two FIFOs.
    •	B: Element-wise add kernel (z[i] = x[i] + y[i]).
    •	C: Consumer prints/stores the result.
•	FIFOs:
    •	of_x, of_y (A→B)
    •	of_z (B→C)
•	Why: Teaches you how to run two input FIFOs into one kernel.
"""

# This is a v2 implementation
# Pulls data from main memory, routes data through shim and mem tiles into worker and back
# Streams data in small chunks to the worker

import aie.iron as iron
import numpy as np
import sys

from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_
from aie.iron.placers import SequentialPlacer

@iron.jit(is_placed=False)
def vec_vec_addition(inA, inB, outC):
    data_size = inA.numel()
    element_type = inA.dtype
    
    shim_tile_size = 128
    worker_tile_size = 32

    # Interface for FIFOs
    data_ty = np.ndarray[(data_size,), np.dtype[element_type]]
    data_shim_ty = np.ndarray[(shim_tile_size,), np.dtype[element_type]]
    data_worker_ty = np.ndarray[(worker_tile_size,), np.dtype[element_type]]

    of_inA_shim = ObjectFifo(data_shim_ty, name="of_inA_shim")
    of_inA = of_inA_shim.cons().forward(obj_type=data_worker_ty, name="of_inA")


    of_inB_shim= ObjectFifo(data_ty, name="of_inB_shim")
    of_inB = of_inB_shim.cons().forward(obj_type=data_worker_ty, name="of_inB")
    
    of_outC_shim = ObjectFifo(data_worker_ty, name="of_outC_shim")
    of_outC = of_outC_shim.cons().forward(obj_type=data_shim_ty, name="of_outC")

    def core_fn(_of_inA, _of_inB, _of_outC):
        elem_inA = _of_inA.acquire(1)
        elem_inB = _of_inB.acquire(1)
        elem_outC = _of_outC.acquire(1)

        for i in range_(worker_tile_size):
            elem_outC[i] = elem_inA[i] + elem_inB[i]

        _of_inA.release(1)
        _of_inB.release(1)
        _of_outC.release(1)

    my_worker = Worker(core_fn, [of_inA.cons(), of_inB.cons(), of_outC_shim.prod()])

    rt = Runtime()
    with rt.sequence(data_ty, data_ty, data_ty) as (inA, inB, outC):
        rt.start(my_worker)
        rt.fill(of_inA_shim.prod(), inA)
        rt.fill(of_inB_shim.prod(), inB)
        rt.drain(of_outC.cons(), outC, wait=True)

    my_program = Program(iron.get_current_device(), rt)

    return my_program.resolve_program(SequentialPlacer())


def main():
    data_size = 16384
    element_type = np.float64
    
    inputA_np = np.random.rand(data_size).astype(element_type)
    inputB_np = np.random.rand(data_size).astype(element_type)
    inputC_np = np.zeros_like(inputA_np)
    
    
    inputA = iron.tensor(inputA_np)
    inputB = iron.tensor(inputB_np)
    outputC = iron.tensor(inputC_np)

    vec_vec_addition(inputA, inputB, outputC)

    # Validate using allclose for floats
    expected = inputA_np + inputB_np
    actual = outputC.numpy()
    if np.allclose(expected, actual, atol=1e-6):
        print("\nPASS!\n")
        sys.exit(0)
    else:
        diff = np.abs(expected - actual)
        print("\nFAIL: max error =", diff.max())
        sys.exit(1)

if __name__ == "__main__":
    main()
