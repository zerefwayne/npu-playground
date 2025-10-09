import sys
import numpy as np
import argparse
import aie.iron as iron

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_

@iron.jit
def external_mem_to_core(input0, output):
    
    @device(iron.get_current_device())
    def device_body():
        
        tile_ty = np.ndarray[(24,), np.dtype[np.int32]]
        
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)
        
        of_in = object_fifo("in", ShimTile, ComputeTile2, 2, tile_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, 2, tile_ty)
        
        
        @core(ComputeTile2)
        def core_body():
            for _ in range_(2):
                elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                
                for i in range_(24):
                    elem_out[i] = elem_in[i] + 1
                    
                of_in.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)
                
        data_ty = np.ndarray[(24,), np.dtype[np.int32]]
        
        @runtime_sequence(data_ty, data_ty)
        def sequence(input0, output):
            npu_dma_memcpy_nd(
                metadata=of_in, bd_id=1, mem=input0, sizes=[1, 1, 1, 24]
            )
            npu_dma_memcpy_nd(
                metadata=of_out, bd_id=0, mem=output, sizes=[1, 1, 1, 24]
            )
            dma_wait(of_out)
            
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
        default=24,
        help="Number of elements (default: 48)",
    )
    args = parser.parse_args()

    # Construct two input random tensors and an output zeroed tensor
    # The three tensor are in memory accessible to the NPU
    input0 = iron.randint(0, 100, (args.num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros(24, dtype=np.int32)
    
    iron.set_current_device(device_map[args.device])
    
    external_mem_to_core(input0, output)
    
    print(input0)
    print(output)
    
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