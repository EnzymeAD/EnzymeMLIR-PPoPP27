module {
  func.func @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<1024x1024xf64>, %arg6: memref<1024x1024xf64>, %arg7: memref<1024x1024xf64>) {
    affine.for %arg8 = 0 to 1024 {
      affine.for %arg9 = 0 to 1024 {
        %0 = affine.load %arg5[%arg8, %arg9] : memref<1024x1024xf64>
        %1 = arith.mulf %0, %arg4 : f64
        affine.store %1, %arg5[%arg8, %arg9] : memref<1024x1024xf64>
        affine.for %arg10 = 0 to 1024 {
          %2 = affine.load %arg6[%arg8, %arg10] : memref<1024x1024xf64>
          %3 = arith.mulf %arg3, %2 : f64
          %4 = affine.load %arg7[%arg10, %arg9] : memref<1024x1024xf64>
          %5 = arith.mulf %3, %4 : f64
          %6 = affine.load %arg5[%arg8, %arg9] : memref<1024x1024xf64>
          %7 = arith.addf %6, %5 : f64
          affine.store %7, %arg5[%arg8, %arg9] : memref<1024x1024xf64>
        }
      }
    }
    return
  }
}

