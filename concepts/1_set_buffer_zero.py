from aie.dialects.aie import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

def mlir_aie_design():
    
    @device(AIEDevice.npu1)
    def device_body():
        
        ComputeTile1 = tile(0, 2)
        
        data_size = 48
        data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]
        
        @core(ComputeTile1)
        def core_body():
            local = buffer(ComputeTile1, data_ty, name="local")
            for i in range_(data_size):
                local[i] = 0
                

with mlir_mod_ctx() as ctx:
    mlir_aie_design()
    print(ctx.module)