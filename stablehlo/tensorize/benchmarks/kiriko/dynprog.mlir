#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  func.func @kernel_dynprog(%arg0: i32, %arg1: i32, %arg2: memref<50x50xi32>, %arg3: memref<50x50xi32>, %arg4: memref<50x50x50xi32>, %arg5: memref<50xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg6 = 0 to 10000 {
      affine.for %arg7 = 0 to 50 {
        affine.for %arg8 = 0 to 50 {
          affine.store %c0_i32, %arg2[%arg7, %arg8] : memref<50x50xi32>
        }
      }
      affine.for %arg7 = 0 to 49 {
        affine.for %arg8 = #map(%arg7) to 50 {
          affine.store %c0_i32, %arg4[%arg7, %arg8, %arg7] : memref<50x50x50xi32>
          affine.for %arg9 = #map(%arg7) to #map1(%arg8) {
            %7 = affine.load %arg4[%arg7, %arg8, %arg9 - 1] : memref<50x50x50xi32>
            %8 = affine.load %arg2[%arg7, %arg9] : memref<50x50xi32>
            %9 = arith.addi %7, %8 : i32
            %10 = affine.load %arg2[%arg9, %arg8] : memref<50x50xi32>
            %11 = arith.addi %9, %10 : i32
            affine.store %11, %arg4[%arg7, %arg8, %arg9] : memref<50x50x50xi32>
          }
          %4 = affine.load %arg4[%arg7, %arg8, %arg8 - 1] : memref<50x50x50xi32>
          %5 = affine.load %arg3[%arg7, %arg8] : memref<50x50xi32>
          %6 = arith.addi %4, %5 : i32
          affine.store %6, %arg2[%arg7, %arg8] : memref<50x50xi32>
        }
      }
      %1 = affine.load %arg2[0, 49] : memref<50x50xi32>
      %2 = affine.load %alloca[] : memref<i32>
      %3 = arith.addi %2, %1 : i32
      affine.store %3, %alloca[] : memref<i32>
    }
    %0 = affine.load %alloca[] : memref<i32>
    affine.store %0, %arg5[0] : memref<50xi32>
    return
  }
}

