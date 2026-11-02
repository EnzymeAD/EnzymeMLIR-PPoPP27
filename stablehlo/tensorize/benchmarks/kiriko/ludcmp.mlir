#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (-d0 + 1024)>
module {
  func.func @kernel_ludcmp(%arg0: i32, %arg1: memref<1025x1025xf64>, %arg2: memref<1025xf64>, %arg3: memref<1025xf64>, %arg4: memref<1025xf64>) {
    %0 = llvm.mlir.undef : f64
    %cst = arith.constant 1.000000e+00 : f64
    %alloca = memref.alloca() : memref<f64>
    affine.store %0, %alloca[] : memref<f64>
    affine.store %cst, %arg2[0] : memref<1025xf64>
    affine.for %arg5 = 0 to 1024 {
      affine.for %arg6 = #map(%arg5) to 1025 {
        %5 = affine.load %arg1[%arg6, %arg5] : memref<1025x1025xf64>
        affine.store %5, %alloca[] : memref<f64>
        affine.for %arg7 = 0 to #map1(%arg5) {
          %9 = affine.load %alloca[] : memref<f64>
          %10 = affine.load %arg1[%arg6, %arg7] : memref<1025x1025xf64>
          %11 = affine.load %arg1[%arg7, %arg5] : memref<1025x1025xf64>
          %12 = arith.mulf %10, %11 : f64
          %13 = arith.subf %9, %12 : f64
          affine.store %13, %alloca[] : memref<f64>
        }
        %6 = affine.load %alloca[] : memref<f64>
        %7 = affine.load %arg1[%arg5, %arg5] : memref<1025x1025xf64>
        %8 = arith.divf %6, %7 : f64
        affine.store %8, %arg1[%arg6, %arg5] : memref<1025x1025xf64>
      }
      affine.for %arg6 = #map(%arg5) to 1025 {
        %5 = affine.load %arg1[%arg5 + 1, %arg6] : memref<1025x1025xf64>
        affine.store %5, %alloca[] : memref<f64>
        affine.for %arg7 = 0 to #map(%arg5) {
          %7 = affine.load %alloca[] : memref<f64>
          %8 = affine.load %arg1[%arg5 + 1, %arg7] : memref<1025x1025xf64>
          %9 = affine.load %arg1[%arg7, %arg6] : memref<1025x1025xf64>
          %10 = arith.mulf %8, %9 : f64
          %11 = arith.subf %7, %10 : f64
          affine.store %11, %alloca[] : memref<f64>
        }
        %6 = affine.load %alloca[] : memref<f64>
        affine.store %6, %arg1[%arg5 + 1, %arg6] : memref<1025x1025xf64>
      }
    }
    %1 = affine.load %arg2[0] : memref<1025xf64>
    affine.store %1, %arg4[0] : memref<1025xf64>
    affine.for %arg5 = 1 to 1025 {
      %5 = affine.load %arg2[%arg5] : memref<1025xf64>
      affine.store %5, %alloca[] : memref<f64>
      affine.for %arg6 = 0 to #map1(%arg5) {
        %7 = affine.load %alloca[] : memref<f64>
        %8 = affine.load %arg1[%arg5, %arg6] : memref<1025x1025xf64>
        %9 = affine.load %arg4[%arg6] : memref<1025xf64>
        %10 = arith.mulf %8, %9 : f64
        %11 = arith.subf %7, %10 : f64
        affine.store %11, %alloca[] : memref<f64>
      }
      %6 = affine.load %alloca[] : memref<f64>
      affine.store %6, %arg4[%arg5] : memref<1025xf64>
    }
    %2 = affine.load %arg4[1024] : memref<1025xf64>
    %3 = affine.load %arg1[1024, 1024] : memref<1025x1025xf64>
    %4 = arith.divf %2, %3 : f64
    affine.store %4, %arg3[1024] : memref<1025xf64>
    affine.for %arg5 = 0 to 1024 {
      %5 = affine.load %arg4[-%arg5 + 1023] : memref<1025xf64>
      affine.store %5, %alloca[] : memref<f64>
      affine.for %arg6 = #map2(%arg5) to 1025 {
        %9 = affine.load %alloca[] : memref<f64>
        %10 = affine.load %arg1[-%arg5 + 1023, %arg6] : memref<1025x1025xf64>
        %11 = affine.load %arg3[%arg6] : memref<1025xf64>
        %12 = arith.mulf %10, %11 : f64
        %13 = arith.subf %9, %12 : f64
        affine.store %13, %alloca[] : memref<f64>
      }
      %6 = affine.load %alloca[] : memref<f64>
      %7 = affine.load %arg1[-%arg5 + 1023, -%arg5 + 1023] : memref<1025x1025xf64>
      %8 = arith.divf %6, %7 : f64
      affine.store %8, %arg3[-%arg5 + 1023] : memref<1025xf64>
    }
    return
  }
}

