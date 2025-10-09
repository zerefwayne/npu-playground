import sys
import numpy as np
import argparse
import aie.iron as iron

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_

@iron.jit
def two_kernels(input0, output):
    
    @device(iron.get_current_device())
    def device_body():
        
        data_ty = np.ndarray[(16,), np.dtype[np.int32]]
        
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile2 = tile(0, 2)
        ComputeTile3 = tile(0, 3)
        
        of_in0 = object_fifo(
            "shim_mem", ShimTile, MemTile, 2, data_ty
        )
        
        of_in1 = object_fifo(
            "mem_compute", MemTile, ComputeTile2, 2, data_ty
        )
        
        object_fifo_link(of_in0, of_in1)
        
        of_in = object_fifo(
            "in", ComputeTile2, ComputeTile3, 2, data_ty
        )
        
        of_out = object_fifo(
            "out", ComputeTile3, ShimTile, 2, data_ty
        )
        
        @core(ComputeTile2)
        def core_body():
            elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
            elem_out = of_in.acquire(ObjectFifoPort.Produce, 1)
            for i in range_(16):
                # Why does elem_out[i] = i not work here?
                elem_out[i] = elem_in[i] // 2
            of_in1.release(ObjectFifoPort.Consume, 1)
            of_in.release(ObjectFifoPort.Produce, 1)
                
        @core(ComputeTile3)
        def core_body():
                elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                
                for i in range_(16):
                    elem_out[i] = elem_in[i] + 3
                
                of_in.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)
                
        @runtime_sequence(data_ty, data_ty)
        def sequence(A, C):
            in_task = shim_dma_single_bd_task(
                of_in0, A, sizes=[1, 1, 1, 16]
            )
            out_task = shim_dma_single_bd_task(
                of_out, C, sizes=[1, 1, 1, 16], issue_token=True
            )
            dma_start_task(in_task, out_task)
            dma_await_task(out_task)
            
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
        default=16,
        help="Number of elements (default: 48)",
    )
    args = parser.parse_args()

    # Construct two input random tensors and an output zeroed tensor
    # The three tensor are in memory accessible to the NPU
    input0 = iron.randint(0, 100, (args.num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros(16, dtype=np.int32)
    
    iron.set_current_device(device_map[args.device])
    
    two_kernels(input0, output)
    
    print(input0)
    print(output)
    
    # Check the correctness of the result
    e = np.equal((input0.numpy() // 2) + 3, output.numpy())
    errors = np.size(e) - np.count_nonzero(e)
    
    # Optionally, print the results
    if args.verbose:
        print(f"{'input0':>4} + {'input1':>4} = {'output':>4}")
        print("-" * 34)
        count = input0.numel()
        for idx, (a, c) in enumerate(
            zip(input0[:count], output[:count])
        ):
            print(f"{idx:2}: ({a:2} // 2) + 3 = {c:4}")
            
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
                
        