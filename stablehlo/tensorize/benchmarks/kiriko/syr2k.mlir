module {
  func.func @kernel_syr2k(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<1024x1024xf64>, %arg5: memref<1024x1024xf64>, %arg6: memref<1024x1024xf64>) {
    affine.for %arg7 = 0 to 1024 {
      affine.for %arg8 = 0 to 1024 {
        %0 = affine.load %arg4[%arg7, %arg8] : memref<1024x1024xf64>
        %1 = arith.mulf %0, %arg3 : f64
        affine.store %1, %arg4[%arg7, %arg8] : memref<1024x1024xf64>
      }
    }
    affine.for %arg7 = 0 to 1024 {
      affine.for %arg8 = 0 to 1024 {
        affine.for %arg9 = 0 to 1024 {
          %0 = affine.load %arg5[%arg7, %arg9] : memref<1024x1024xf64>
          %1 = arith.mulf %arg2, %0 : f64
          %2 = affine.load %arg6[%arg8, %arg9] : memref<1024x1024xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg4[%arg7, %arg8] : memref<1024x1024xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg4[%arg7, %arg8] : memref<1024x1024xf64>
          %6 = affine.load %arg6[%arg7, %arg9] : memref<1024x1024xf64>
          %7 = arith.mulf %arg2, %6 : f64
          %8 = affine.load %arg5[%arg8, %arg9] : memref<1024x1024xf64>
          %9 = arith.mulf %7, %8 : f64
          %10 = arith.addf %5, %9 : f64
          affine.store %10, %arg4[%arg7, %arg8] : memref<1024x1024xf64>
        }
      }
    }
    return
  }
}

