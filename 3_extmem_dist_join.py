import numpy as np
import sys
import argparse
import aie.iron as iron

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import (
    _for as range_,
)

n_cores = 3
buffer_depth = 2
data_size = 48
tile_size = data_size // n_cores

@iron.jit
def datamov_ext_mem_multi(input0, output0):
    
    @device(iron.get_current_device())
    def device_body():
        data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]
        tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
        
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTiles = [tile(0, 2 + i) for i in range(n_cores)]
        
        inX_fifos = []
        
        of_in = object_fifo("in", ShimTile, MemTile, buffer_depth, data_ty)
        for i in range(n_cores):
            inX_fifos.append(
                object_fifo(
                    f"in{i}",
                    MemTile,
                    ComputeTiles[i],
                    buffer_depth,
                    tile_ty
                )
            )
            
        if n_cores > 1:
            of_offsets = [tile_size * i for i in range(n_cores)]
        else:
            of_offsets = []
        object_fifo_link(of_in, inX_fifos, [], of_offsets)
        
        outX_fifos = []
        
        of_out = object_fifo("out", MemTile, ShimTile, buffer_depth, data_ty)
        for i in range(n_cores):
            outX_fifos.append(
                object_fifo(
                    f"out{i}",
                    ComputeTiles[i],
                    MemTile,
                    buffer_depth,
                    tile_ty
                )
            )
        object_fifo_link(outX_fifos, of_out, of_offsets, [])
        
        for i in range(n_cores):
            @core(ComputeTiles[i])
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    elem_in = inX_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                    elem_out = outX_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                    
                    for j in range_(tile_size):
                        elem_out[j] = elem_in[j] + 1
                        
                    inX_fifos[i].release(ObjectFifoPort.Consume, 1)
                    outX_fifos[i].release(ObjectFifoPort.Produce, 1)
                    
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
    
    datamov_ext_mem_multi(input0, output)
    
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