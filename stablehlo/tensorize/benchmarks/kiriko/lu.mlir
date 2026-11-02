#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @kernel_lu(%arg0: i32, %arg1: memref<1024x1024xf64>) {
    affine.for %arg2 = 0 to 1024 {
      affine.for %arg3 = #map(%arg2) to 1024 {
        %0 = affine.load %arg1[%arg2, %arg3] : memref<1024x1024xf64>
        %1 = affine.load %arg1[%arg2, %arg2] : memref<1024x1024xf64>
        %2 = arith.divf %0, %1 : f64
        affine.store %2, %arg1[%arg2, %arg3] : memref<1024x1024xf64>
      }
      affine.for %arg3 = #map(%arg2) to 1024 {
        affine.for %arg4 = #map(%arg2) to 1024 {
          %0 = affine.load %arg1[%arg3, %arg4] : memref<1024x1024xf64>
          %1 = affine.load %arg1[%arg3, %arg2] : memref<1024x1024xf64>
          %2 = affine.load %arg1[%arg2, %arg4] : memref<1024x1024xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = arith.subf %0, %3 : f64
          affine.store %4, %arg1[%arg3, %arg4] : memref<1024x1024xf64>
        }
      }
    }
    return
  }
}

