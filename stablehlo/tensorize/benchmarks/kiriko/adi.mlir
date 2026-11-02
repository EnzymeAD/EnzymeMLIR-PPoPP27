module {
  func.func @kernel_adi(%arg0: i32, %arg1: i32, %arg2: memref<1024x1024xf64>, %arg3: memref<1024x1024xf64>, %arg4: memref<1024x1024xf64>) {
    affine.for %arg5 = 0 to 50 {
      affine.for %arg6 = 0 to 1024 {
        affine.for %arg7 = 1 to 1024 {
          %0 = affine.load %arg2[%arg6, %arg7] : memref<1024x1024xf64>
          %1 = affine.load %arg2[%arg6, %arg7 - 1] : memref<1024x1024xf64>
          %2 = affine.load %arg3[%arg6, %arg7] : memref<1024x1024xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg4[%arg6, %arg7 - 1] : memref<1024x1024xf64>
          %5 = arith.divf %3, %4 : f64
          %6 = arith.subf %0, %5 : f64
          affine.store %6, %arg2[%arg6, %arg7] : memref<1024x1024xf64>
          %7 = affine.load %arg4[%arg6, %arg7] : memref<1024x1024xf64>
          %8 = affine.load %arg3[%arg6, %arg7] : memref<1024x1024xf64>
          %9 = arith.mulf %8, %8 : f64
          %10 = affine.load %arg4[%arg6, %arg7 - 1] : memref<1024x1024xf64>
          %11 = arith.divf %9, %10 : f64
          %12 = arith.subf %7, %11 : f64
          affine.store %12, %arg4[%arg6, %arg7] : memref<1024x1024xf64>
        }
      }
      affine.for %arg6 = 0 to 1024 {
        %0 = affine.load %arg2[%arg6, 1023] : memref<1024x1024xf64>
        %1 = affine.load %arg4[%arg6, 1023] : memref<1024x1024xf64>
        %2 = arith.divf %0, %1 : f64
        affine.store %2, %arg2[%arg6, 1023] : memref<1024x1024xf64>
      }
      affine.for %arg6 = 0 to 1024 {
        affine.for %arg7 = 0 to 1022 {
          %0 = affine.load %arg2[%arg6, -%arg7 + 1022] : memref<1024x1024xf64>
          %1 = affine.load %arg2[%arg6, -%arg7 + 1021] : memref<1024x1024xf64>
          %2 = affine.load %arg3[%arg6, -%arg7 + 1021] : memref<1024x1024xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = arith.subf %0, %3 : f64
          %5 = affine.load %arg4[%arg6, -%arg7 + 1021] : memref<1024x1024xf64>
          %6 = arith.divf %4, %5 : f64
          affine.store %6, %arg2[%arg6, -%arg7 + 1022] : memref<1024x1024xf64>
        }
      }
      affine.for %arg6 = 1 to 1024 {
        affine.for %arg7 = 0 to 1024 {
          %0 = affine.load %arg2[%arg6, %arg7] : memref<1024x1024xf64>
          %1 = affine.load %arg2[%arg6 - 1, %arg7] : memref<1024x1024xf64>
          %2 = affine.load %arg3[%arg6, %arg7] : memref<1024x1024xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %arg4[%arg6 - 1, %arg7] : memref<1024x1024xf64>
          %5 = arith.divf %3, %4 : f64
          %6 = arith.subf %0, %5 : f64
          affine.store %6, %arg2[%arg6, %arg7] : memref<1024x1024xf64>
          %7 = affine.load %arg4[%arg6, %arg7] : memref<1024x1024xf64>
          %8 = affine.load %arg3[%arg6, %arg7] : memref<1024x1024xf64>
          %9 = arith.mulf %8, %8 : f64
          %10 = affine.load %arg4[%arg6 - 1, %arg7] : memref<1024x1024xf64>
          %11 = arith.divf %9, %10 : f64
          %12 = arith.subf %7, %11 : f64
          affine.store %12, %arg4[%arg6, %arg7] : memref<1024x1024xf64>
        }
      }
      affine.for %arg6 = 0 to 1024 {
        %0 = affine.load %arg2[1023, %arg6] : memref<1024x1024xf64>
        %1 = affine.load %arg4[1023, %arg6] : memref<1024x1024xf64>
        %2 = arith.divf %0, %1 : f64
        affine.store %2, %arg2[1023, %arg6] : memref<1024x1024xf64>
      }
      affine.for %arg6 = 0 to 1022 {
        affine.for %arg7 = 0 to 1024 {
          %0 = affine.load %arg2[-%arg6 + 1022, %arg7] : memref<1024x1024xf64>
          %1 = affine.load %arg2[-%arg6 + 1021, %arg7] : memref<1024x1024xf64>
          %2 = affine.load %arg3[-%arg6 + 1021, %arg7] : memref<1024x1024xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = arith.subf %0, %3 : f64
          %5 = affine.load %arg4[-%arg6 + 1022, %arg7] : memref<1024x1024xf64>
          %6 = arith.divf %4, %5 : f64
          affine.store %6, %arg2[-%arg6 + 1022, %arg7] : memref<1024x1024xf64>
        }
      }
    }
    return
  }
}

