import numpy as np
import sys
import argparse
from aie.dialects.aie import *
from aie.extras.context import mlir_mod_ctx
import aie.iron as iron

from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import (
    _for as range_,
)

dev = AIEDevice.npu1_1col

@iron.jit
def mlir_aie_design(input0, output0):
        buffer_depth = 2
        data_size = 48
        
        @device(iron.get_current_device())
        def device_body():
            data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]
            
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile = tile(0, 2)
            
            of_in = object_fifo("in", ShimTile, MemTile, buffer_depth, data_ty)
            of_in1 = object_fifo("in1", MemTile, ComputeTile, buffer_depth, data_ty)
            object_fifo_link(of_in, of_in1)
            
            of_out1 = object_fifo("out1", ComputeTile, MemTile, buffer_depth, data_ty)
            of_out = object_fifo("out", MemTile, ShimTile, buffer_depth, data_ty)
            object_fifo_link(of_out1, of_out)
            
            @core(ComputeTile)
            def core_body():
                for _ in range_(sys.maxsize):
                    elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                    
                    for i in range_(data_size):
                        elem_out[i] = elem_in[i] + 1
                        
                    of_in1.release(ObjectFifoPort.Consume, 1)
                    of_out1.release(ObjectFifoPort.Produce, 1)
                    
            @runtime_sequence(data_ty, data_ty)
            def sequence(A, B):
                in_task = shim_dma_single_bd_task(of_in, A, sizes=[1, 1, 1, data_size])
                out_task = shim_dma_single_bd_task(of_out, B, sizes=[1, 1, 1, data_size], issue_token=True)
                
                dma_start_task(in_task, out_task)
                dma_await_task(out_task)
                dma_free_task(in_task)
                
                
def main():
    device_map = {
        "npu": AIEDevice.npu1_1col,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["npu", "npu2", "xcvc1902"],
        default="npu",
        help="Target device",
    )
    parser.add_argument(
        "-n",
        "--num-elements",
        type=int,
        default=48,
        help="Number of elements (default: 48)",
    )
    args = parser.parse_args()

    # Construct two input random tensors and an output zeroed tensor
    # The three tensor are in memory accessible to the NPU
    input0 = iron.randint(0, 100, (args.num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input0)
    
    iron.set_current_device(device_map[args.device])
    
    mlir_aie_design(input0, output)
    
    # Check the correctness of the result
    e = np.equal(input0.numpy() + 1, output.numpy())
    errors = np.size(e) - np.count_nonzero(e)
    
    # Optionally, print the results
    if args.verbose:
        print(f"{'input0':>4} + {'input1':>4} = {'output':>4}")
        print("-" * 34)
        count = input0.numel()
        for idx, (a, c) in enumerate(
            zip(input0[:count], output[:count])
        ):
            print(f"{idx:2}: {a:4} + 1 = {c:4}")
            
    # If the result is correct, exit with a success code.
    # Otherwise, exit with a failure code
    if not errors:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        sys.exit(-1)
        
if __name__ == "__main__":
    main()
    
    