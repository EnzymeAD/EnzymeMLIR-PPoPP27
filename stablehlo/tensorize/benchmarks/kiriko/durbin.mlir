#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel_durbin(%arg0: i32, %arg1: memref<4000x4000xf64>, %arg2: memref<4000x4000xf64>, %arg3: memref<4000xf64>, %arg4: memref<4000xf64>, %arg5: memref<4000xf64>, %arg6: memref<4000xf64>) {
    %cst = arith.constant 1.000000e+00 : f64
    %0 = affine.load %arg5[0] : memref<4000xf64>
    affine.store %0, %arg1[0, 0] : memref<4000x4000xf64>
    affine.store %cst, %arg4[0] : memref<4000xf64>
    %1 = affine.load %arg5[0] : memref<4000xf64>
    affine.store %1, %arg3[0] : memref<4000xf64>
    affine.for %arg7 = 1 to 4000 {
      %2 = affine.load %arg4[%arg7 - 1] : memref<4000xf64>
      %3 = affine.load %arg3[%arg7 - 1] : memref<4000xf64>
      %4 = arith.mulf %3, %3 : f64
      %5 = arith.mulf %4, %2 : f64
      %6 = arith.subf %2, %5 : f64
      affine.store %6, %arg4[%arg7] : memref<4000xf64>
      %7 = affine.load %arg5[%arg7] : memref<4000xf64>
      affine.store %7, %arg2[0, %arg7] : memref<4000x4000xf64>
      affine.for %arg8 = 0 to #map(%arg7) {
        %13 = affine.load %arg2[%arg8, %arg7] : memref<4000x4000xf64>
        %14 = affine.load %arg5[%arg7 - %arg8 - 1] : memref<4000xf64>
        %15 = affine.load %arg1[%arg8, %arg7 - 1] : memref<4000x4000xf64>
        %16 = arith.mulf %14, %15 : f64
        %17 = arith.addf %13, %16 : f64
        affine.store %17, %arg2[%arg8 + 1, %arg7] : memref<4000x4000xf64>
      }
      %8 = affine.load %arg2[%arg7, %arg7] : memref<4000x4000xf64>
      %9 = arith.negf %8 : f64
      %10 = affine.load %arg4[%arg7] : memref<4000xf64>
      %11 = arith.mulf %9, %10 : f64
      affine.store %11, %arg3[%arg7] : memref<4000xf64>
      affine.for %arg8 = 0 to #map(%arg7) {
        %13 = affine.load %arg1[%arg8, %arg7 - 1] : memref<4000x4000xf64>
        %14 = affine.load %arg3[%arg7] : memref<4000xf64>
        %15 = affine.load %arg1[%arg7 - %arg8 - 1, %arg7 - 1] : memref<4000x4000xf64>
        %16 = arith.mulf %14, %15 : f64
        %17 = arith.addf %13, %16 : f64
        affine.store %17, %arg1[%arg8, %arg7] : memref<4000x4000xf64>
      }
      %12 = affine.load %arg3[%arg7] : memref<4000xf64>
      affine.store %12, %arg1[%arg7, %arg7] : memref<4000x4000xf64>
    }
    affine.for %arg7 = 0 to 4000 {
      %2 = affine.load %arg1[%arg7, 3999] : memref<4000x4000xf64>
      affine.store %2, %arg6[%arg7] : memref<4000xf64>
    }
    return
  }
}

