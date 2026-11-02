#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel_reg_detect(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<6x6xi32>, %arg4: memref<6x6xi32>, %arg5: memref<6x6xi32>, %arg6: memref<6x6x64xi32>, %arg7: memref<6x6x64xi32>) {
    affine.for %arg8 = 0 to 10000 {
      affine.for %arg9 = 0 to 6 {
        affine.for %arg10 = #map(%arg9) to 6 {
          affine.for %arg11 = 0 to 64 {
            %0 = affine.load %arg3[%arg9, %arg10] : memref<6x6xi32>
            affine.store %0, %arg6[%arg9, %arg10, %arg11] : memref<6x6x64xi32>
          }
        }
      }
      affine.for %arg9 = 0 to 6 {
        affine.for %arg10 = #map(%arg9) to 6 {
          %0 = affine.load %arg6[%arg9, %arg10, 0] : memref<6x6x64xi32>
          affine.store %0, %arg7[%arg9, %arg10, 0] : memref<6x6x64xi32>
          affine.for %arg11 = 1 to 64 {
            %2 = affine.load %arg7[%arg9, %arg10, %arg11 - 1] : memref<6x6x64xi32>
            %3 = affine.load %arg6[%arg9, %arg10, %arg11] : memref<6x6x64xi32>
            %4 = arith.addi %2, %3 : i32
            affine.store %4, %arg7[%arg9, %arg10, %arg11] : memref<6x6x64xi32>
          }
          %1 = affine.load %arg7[%arg9, %arg10, 63] : memref<6x6x64xi32>
          affine.store %1, %arg4[%arg9, %arg10] : memref<6x6xi32>
        }
      }
      affine.for %arg9 = 0 to 6 {
        %0 = affine.load %arg4[0, %arg9] : memref<6x6xi32>
        affine.store %0, %arg5[0, %arg9] : memref<6x6xi32>
      }
      affine.for %arg9 = 1 to 6 {
        affine.for %arg10 = #map(%arg9) to 6 {
          %0 = affine.load %arg5[%arg9 - 1, %arg10 - 1] : memref<6x6xi32>
          %1 = affine.load %arg4[%arg9, %arg10] : memref<6x6xi32>
          %2 = arith.addi %0, %1 : i32
          affine.store %2, %arg5[%arg9, %arg10] : memref<6x6xi32>
        }
      }
    }
    return
  }
}

