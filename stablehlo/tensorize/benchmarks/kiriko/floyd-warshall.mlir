module {
  func.func @kernel_floyd_warshall(%arg0: i32, %arg1: memref<1024x1024xf64>) {
    affine.for %arg2 = 0 to 1024 {
      affine.for %arg3 = 0 to 1024 {
        affine.for %arg4 = 0 to 1024 {
          %0 = affine.load %arg1[%arg3, %arg4] : memref<1024x1024xf64>
          %1 = affine.load %arg1[%arg3, %arg2] : memref<1024x1024xf64>
          %2 = affine.load %arg1[%arg2, %arg4] : memref<1024x1024xf64>
          %3 = arith.addf %1, %2 : f64
          %4 = arith.cmpf olt, %0, %3 : f64
          %5 = arith.select %4, %0, %3 : f64
          affine.store %5, %arg1[%arg3, %arg4] : memref<1024x1024xf64>
        }
      }
    }
    return
  }
}

