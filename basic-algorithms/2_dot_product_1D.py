import numpy as np
import sys
import argparse
import aie.iron as iron

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_

@iron.jit
def dot_product_1D(input0, input1, output):
    
    @device(iron.get_current_device())
    def device_body():
        # Not sure why, just wanted to try some different sizes
        # TODO: Find out what's a good balance to have
        data_tile_size = 4096
        shim_tile_size = 512
        core_tile_size = 32
        output_tile_size = 1
        
        data_ty = np.ndarray[(data_tile_size,), np.dtype[np.int32]]
        shim_tile_ty = np.ndarray[(shim_tile_size,), np.dtype[np.int32]]
        core_tile_ty = np.ndarray[(core_tile_size,), np.dtype[np.int32]]
        output_tile_ty = np.ndarray[(output_tile_size,), np.dtype[np.int32]]
        
        ShimTile = tile(0, 0)
        MemTile = tile(0, 1)
        ComputeTile2 = tile(0, 2)
        ComputeTile3 = tile(0, 3)
        
        of_in0_0 = object_fifo("of_in0_0", ShimTile, MemTile, 2, shim_tile_ty)
        of_in0_1 = object_fifo("of_in0_1", MemTile, ComputeTile2, 2, core_tile_ty)
        object_fifo_link(of_in0_0, of_in0_1)
        
        of_in1_0 = object_fifo("of_in1_0", ShimTile, MemTile, 2, shim_tile_ty)
        of_in1_1 = object_fifo("of_in1_1", MemTile, ComputeTile2, 2, core_tile_ty)
        object_fifo_link(of_in1_0, of_in1_1)
        
        of_2_3 = object_fifo("of_2_3", ComputeTile2, ComputeTile3, 2, core_tile_ty)
        
        of_out1 = object_fifo("of_out1", ComputeTile3, MemTile, 1, output_tile_ty)
        of_out0 = object_fifo("of_out0", MemTile, ShimTile, 1, output_tile_ty)
        object_fifo_link(of_out1, of_out0)
        
        data_buffer = buffer(ComputeTile3, output_tile_ty, "sum_result_tile3")
        
        @core(ComputeTile2)
        def core_body():
            for _ in range_(data_tile_size // core_tile_size):
                elem_in0 = of_in0_1.acquire(ObjectFifoPort.Consume, 1)
                elem_in1 = of_in1_1.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_2_3.acquire(ObjectFifoPort.Produce, 1)
                
                for i in range_(core_tile_size):
                    elem_out[i] = elem_in0[i] * elem_in1[i]
                    
                of_in0_1.release(ObjectFifoPort.Consume, 1)
                of_in1_1.release(ObjectFifoPort.Consume, 1)
                of_2_3.release(ObjectFifoPort.Produce, 1)
                
        @core(ComputeTile3)
        def core_body():
            data_buffer[0] = 0
            elem_res = of_out1.acquire(ObjectFifoPort.Produce, 1)
            
            for _ in range_(data_tile_size // core_tile_size):
                elem_in = of_2_3.acquire(ObjectFifoPort.Consume, 1)
                
                for i in range_(core_tile_size):
                    data_buffer[0] = data_buffer[0] + elem_in[i]
                    
                of_2_3.release(ObjectFifoPort.Consume, 1)
                
            elem_res[0] = data_buffer[0]
            of_out1.release(ObjectFifoPort.Produce, 1)
            
        @runtime_sequence(data_ty, data_ty, data_ty)
        def sequence(input0, input1, output):
            npu_dma_memcpy_nd(
                metadata=of_in0_0, bd_id=2, mem=input0, sizes=[1, 1, 1, data_tile_size]
            )
            npu_dma_memcpy_nd(
                metadata=of_in1_0, bd_id=1, mem=input1, sizes=[1, 1, 1, data_tile_size]
            )
            npu_dma_memcpy_nd(
                metadata=of_out0, bd_id=0, mem=output, sizes=[1, 1, 1, output_tile_size]
            )
            dma_wait(of_out0)
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    data_size = 4096
    
    # Construct two input random tensors and an output zeroed tensor
    # The three tensor are in memory accessible to the NPU
    input0 = iron.arange(0, data_size, dtype=np.int32, device="npu")
    input1 = iron.arange(0, data_size, dtype=np.int32, device="npu")
    output = iron.zeros(1, dtype=np.int32)
    
    iron.set_current_device(AIEDevice.npu1_1col)
    
    dot_product_1D(input0, input1, output)
    
    # Check the correctness of the result
    e = np.dot(input0.numpy(), input1.numpy()) == output.numpy()[0]
    
    print("Expected output:", np.dot(input0.numpy(), input1.numpy()))
    print("Actual output:", output.numpy()[0])
    print()
    if e:
        print("Dot product PASSED: Output matches expected result.")
    else:
        print("Dot product FAILED: Output does not match expected result.")
        
if __name__ == "__main__":
    main()
