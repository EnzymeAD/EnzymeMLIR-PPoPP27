module {
  func.func @kernel_doitgen(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<128x128x128xf64>, %arg4: memref<128x128xf64>, %arg5: memref<128x128x128xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg6 = 0 to 128 {
      affine.for %arg7 = 0 to 128 {
        affine.for %arg8 = 0 to 128 {
          affine.store %cst, %arg5[%arg6, %arg7, %arg8] : memref<128x128x128xf64>
          affine.for %arg9 = 0 to 128 {
            %0 = affine.load %arg5[%arg6, %arg7, %arg8] : memref<128x128x128xf64>
            %1 = affine.load %arg3[%arg6, %arg7, %arg9] : memref<128x128x128xf64>
            %2 = affine.load %arg4[%arg9, %arg8] : memref<128x128xf64>
            %3 = arith.mulf %1, %2 : f64
            %4 = arith.addf %0, %3 : f64
            affine.store %4, %arg5[%arg6, %arg7, %arg8] : memref<128x128x128xf64>
          }
        }
        affine.for %arg8 = 0 to 128 {
          %0 = affine.load %arg5[%arg6, %arg7, %arg8] : memref<128x128x128xf64>
          affine.store %0, %arg3[%arg6, %arg7, %arg8] : memref<128x128x128xf64>
        }
      }
    }
    return
  }
}

