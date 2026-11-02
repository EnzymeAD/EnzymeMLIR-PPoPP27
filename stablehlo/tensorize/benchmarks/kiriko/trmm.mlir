#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel_trmm(%arg0: i32, %arg1: f64, %arg2: memref<1024x1024xf64>, %arg3: memref<1024x1024xf64>) {
    affine.for %arg4 = 1 to 1024 {
      affine.for %arg5 = 0 to 1024 {
        affine.for %arg6 = 0 to #map(%arg4) {
          %0 = affine.load %arg2[%arg4, %arg6] : memref<1024x1024xf64>
          %1 = arith.mulf %arg1, %0 : f64
          %2 = affine.load %arg3[%arg5, %arg6] : memref<1024x1024xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg3[%arg4, %arg5] : memref<1024x1024xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg3[%arg4, %arg5] : memref<1024x1024xf64>
        }
      }
    }
    return
  }
}

