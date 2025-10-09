import numpy as np
import sys
import argparse
import aie.iron as iron

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_

@iron.jit
def external_L2_core(input0, output):
    
    @device(iron.get_current_device())
    def device_body():
        tile24_ty = np.ndarray[(24,), np.dtype[np.int32]]
        tile8_ty = np.ndarray[(8,), np.dtype[np.int32]]
        
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile2 = tile(0, 2)
        ComputeTile3 = tile(0, 3)
        
        of_in0 = object_fifo("in0", ShimTile, MemTile, 2, tile24_ty)
        of_in1 = object_fifo("in1", MemTile, ComputeTile2, 2, tile8_ty)
        object_fifo_link(of_in0, of_in1)
        
        of_2_3 = object_fifo("in2", ComputeTile2, ComputeTile3, 2, tile8_ty)
        
        of_out1 = object_fifo("out1", ComputeTile3, MemTile, 2, tile8_ty)
        of_out0 = object_fifo("out0", MemTile, ShimTile, 2, tile24_ty)
        object_fifo_link(of_out1, of_out0)
        
        @core(ComputeTile2)
        def core_body():
            for _ in range_(6):
                elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_2_3.acquire(ObjectFifoPort.Produce, 1)
                
                for i in range_(8):
                    elem_out[i] = elem_in[i] + 1
                
                of_in1.release(ObjectFifoPort.Consume, 1)
                of_2_3.release(ObjectFifoPort.Produce, 1)
                
        @core(ComputeTile3)
        def core_body():
            for _ in range_(6):
                elem_in = of_2_3.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
                
                for i in range_(8):
                    elem_out[i] = elem_in[i] * 3
                
                of_2_3.release(ObjectFifoPort.Consume, 1)
                of_out1.release(ObjectFifoPort.Produce, 1)
                
        data_ty = np.ndarray[(48,), np.dtype[np.int32]]
        
        @runtime_sequence(data_ty, data_ty)
        def sequence(input0, output):
            npu_dma_memcpy_nd(
                metadata=of_in0, bd_id=1, mem=input0, sizes=[1, 1, 1, 48]
            )
            npu_dma_memcpy_nd(
                metadata=of_out0, bd_id=0, mem=output, sizes=[1, 1, 1, 48]
            )
            dma_wait(of_out0)
            
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
    
    external_L2_core(input0, output)
    
    print(input0)
    print(output)
    
    # Check the correctness of the result
    e = np.equal((input0.numpy() + 1) * 3, output.numpy())
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