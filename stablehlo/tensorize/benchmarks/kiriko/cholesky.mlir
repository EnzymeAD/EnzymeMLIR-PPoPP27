#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @kernel_cholesky(%arg0: i32, %arg1: memref<1024xf64>, %arg2: memref<1024x1024xf64>) {
    %0 = llvm.mlir.undef : f64
    %cst = arith.constant 1.000000e+00 : f64
    %alloca = memref.alloca() : memref<f64>
    affine.store %0, %alloca[] : memref<f64>
    affine.for %arg3 = 0 to 1024 {
      %1 = affine.load %arg2[%arg3, %arg3] : memref<1024x1024xf64>
      affine.store %1, %alloca[] : memref<f64>
      affine.for %arg4 = 0 to #map(%arg3) {
        %5 = affine.load %alloca[] : memref<f64>
        %6 = affine.load %arg2[%arg3, %arg4] : memref<1024x1024xf64>
        %7 = arith.mulf %6, %6 : f64
        %8 = arith.subf %5, %7 : f64
        affine.store %8, %alloca[] : memref<f64>
      }
      %2 = affine.load %alloca[] : memref<f64>
      %3 = math.sqrt %2 : f64
      %4 = arith.divf %cst, %3 : f64
      affine.store %4, %arg1[%arg3] : memref<1024xf64>
      affine.for %arg4 = #map1(%arg3) to 1024 {
        %5 = affine.load %arg2[%arg3, %arg4] : memref<1024x1024xf64>
        affine.store %5, %alloca[] : memref<f64>
        affine.for %arg5 = 0 to #map(%arg3) {
          %9 = affine.load %alloca[] : memref<f64>
          %10 = affine.load %arg2[%arg4, %arg5] : memref<1024x1024xf64>
          %11 = affine.load %arg2[%arg3, %arg5] : memref<1024x1024xf64>
          %12 = arith.mulf %10, %11 : f64
          %13 = arith.subf %9, %12 : f64
          affine.store %13, %alloca[] : memref<f64>
        }
        %6 = affine.load %alloca[] : memref<f64>
        %7 = affine.load %arg1[%arg3] : memref<1024xf64>
        %8 = arith.mulf %6, %7 : f64
        affine.store %8, %arg2[%arg4, %arg3] : memref<1024x1024xf64>
      }
    }
    return
  }
}

