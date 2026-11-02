module {
  func.func @kernel_2mm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<1024x1024xf64>, %arg7: memref<1024x1024xf64>, %arg8: memref<1024x1024xf64>, %arg9: memref<1024x1024xf64>, %arg10: memref<1024x1024xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg11 = 0 to 1024 {
      affine.for %arg12 = 0 to 1024 {
        affine.store %cst, %arg6[%arg11, %arg12] : memref<1024x1024xf64>
        affine.for %arg13 = 0 to 1024 {
          %0 = affine.load %arg7[%arg11, %arg13] : memref<1024x1024xf64>
          %1 = arith.mulf %arg4, %0 : f64
          %2 = affine.load %arg8[%arg13, %arg12] : memref<1024x1024xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg6[%arg11, %arg12] : memref<1024x1024xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg6[%arg11, %arg12] : memref<1024x1024xf64>
        }
      }
    }
    affine.for %arg11 = 0 to 1024 {
      affine.for %arg12 = 0 to 1024 {
        %0 = affine.load %arg10[%arg11, %arg12] : memref<1024x1024xf64>
        %1 = arith.mulf %0, %arg5 : f64
        affine.store %1, %arg10[%arg11, %arg12] : memref<1024x1024xf64>
        affine.for %arg13 = 0 to 1024 {
          %2 = affine.load %arg6[%arg11, %arg13] : memref<1024x1024xf64>
          %3 = affine.load %arg9[%arg13, %arg12] : memref<1024x1024xf64>
          %4 = arith.mulf %2, %3 : f64
          %5 = affine.load %arg10[%arg11, %arg12] : memref<1024x1024xf64>
          %6 = arith.addf %5, %4 : f64
          affine.store %6, %arg10[%arg11, %arg12] : memref<1024x1024xf64>
        }
      }
    }
    return
  }
}

