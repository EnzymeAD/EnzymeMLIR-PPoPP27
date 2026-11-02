#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<1000x1000xf64>, %arg4: memref<1000x1000xf64>, %arg5: memref<1000xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg6 = 0 to 1000 {
      affine.store %cst, %arg5[%arg6] : memref<1000xf64>
      affine.for %arg7 = 0 to 1000 {
        %2 = affine.load %arg3[%arg7, %arg6] : memref<1000x1000xf64>
        %3 = affine.load %arg5[%arg6] : memref<1000xf64>
        %4 = arith.addf %3, %2 : f64
        affine.store %4, %arg5[%arg6] : memref<1000xf64>
      }
      %0 = affine.load %arg5[%arg6] : memref<1000xf64>
      %1 = arith.divf %0, %arg2 : f64
      affine.store %1, %arg5[%arg6] : memref<1000xf64>
    }
    affine.for %arg6 = 0 to 1000 {
      affine.for %arg7 = 0 to 1000 {
        %0 = affine.load %arg5[%arg7] : memref<1000xf64>
        %1 = affine.load %arg3[%arg6, %arg7] : memref<1000x1000xf64>
        %2 = arith.subf %1, %0 : f64
        affine.store %2, %arg3[%arg6, %arg7] : memref<1000x1000xf64>
      }
    }
    affine.for %arg6 = 0 to 1000 {
      affine.for %arg7 = #map(%arg6) to 1000 {
        affine.store %cst, %arg4[%arg6, %arg7] : memref<1000x1000xf64>
        affine.for %arg8 = 0 to 1000 {
          %1 = affine.load %arg3[%arg8, %arg6] : memref<1000x1000xf64>
          %2 = affine.load %arg3[%arg8, %arg7] : memref<1000x1000xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg4[%arg6, %arg7] : memref<1000x1000xf64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %arg4[%arg6, %arg7] : memref<1000x1000xf64>
        }
        %0 = affine.load %arg4[%arg6, %arg7] : memref<1000x1000xf64>
        affine.store %0, %arg4[%arg7, %arg6] : memref<1000x1000xf64>
      }
    }
    return
  }
}

