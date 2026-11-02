module {
  func.func @kernel_fdtd_apml(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<257x257xf64>, %arg6: memref<257x257xf64>, %arg7: memref<257x257xf64>, %arg8: memref<257x257xf64>, %arg9: memref<257x257x257xf64>, %arg10: memref<257x257x257xf64>, %arg11: memref<257x257x257xf64>, %arg12: memref<257x257x257xf64>, %arg13: memref<257xf64>, %arg14: memref<257xf64>, %arg15: memref<257xf64>, %arg16: memref<257xf64>, %arg17: memref<257xf64>, %arg18: memref<257xf64>) {
    affine.for %arg19 = 0 to 256 {
      affine.for %arg20 = 0 to 256 {
        affine.for %arg21 = 0 to 256 {
          %64 = affine.load %arg10[%arg19, %arg20, %arg21] : memref<257x257x257xf64>
          %65 = affine.load %arg10[%arg19, %arg20 + 1, %arg21] : memref<257x257x257xf64>
          %66 = arith.subf %64, %65 : f64
          %67 = affine.load %arg11[%arg19, %arg20, %arg21 + 1] : memref<257x257x257xf64>
          %68 = arith.addf %66, %67 : f64
          %69 = affine.load %arg11[%arg19, %arg20, %arg21] : memref<257x257x257xf64>
          %70 = arith.subf %68, %69 : f64
          affine.store %70, %arg7[%arg19, %arg20] : memref<257x257xf64>
          %71 = affine.load %arg17[%arg20] : memref<257xf64>
          %72 = affine.load %arg18[%arg20] : memref<257xf64>
          %73 = arith.divf %71, %72 : f64
          %74 = affine.load %arg9[%arg19, %arg20, %arg21] : memref<257x257x257xf64>
          %75 = arith.mulf %73, %74 : f64
          %76 = arith.divf %arg4, %72 : f64
          %77 = arith.mulf %76, %70 : f64
          %78 = arith.subf %75, %77 : f64
          affine.store %78, %arg8[%arg19, %arg20] : memref<257x257xf64>
          %79 = affine.load %arg15[%arg21] : memref<257xf64>
          %80 = affine.load %arg16[%arg21] : memref<257xf64>
          %81 = arith.divf %79, %80 : f64
          %82 = affine.load %arg12[%arg19, %arg20, %arg21] : memref<257x257x257xf64>
          %83 = arith.mulf %81, %82 : f64
          %84 = affine.load %arg14[%arg19] : memref<257xf64>
          %85 = arith.mulf %arg3, %84 : f64
          %86 = arith.divf %85, %80 : f64
          %87 = arith.mulf %86, %78 : f64
          %88 = arith.addf %83, %87 : f64
          %89 = affine.load %arg13[%arg19] : memref<257xf64>
          %90 = arith.mulf %arg3, %89 : f64
          %91 = arith.divf %90, %80 : f64
          %92 = affine.load %arg9[%arg19, %arg20, %arg21] : memref<257x257x257xf64>
          %93 = arith.mulf %91, %92 : f64
          %94 = arith.subf %88, %93 : f64
          affine.store %94, %arg12[%arg19, %arg20, %arg21] : memref<257x257x257xf64>
          %95 = affine.load %arg8[%arg19, %arg20] : memref<257x257xf64>
          affine.store %95, %arg9[%arg19, %arg20, %arg21] : memref<257x257x257xf64>
        }
        %0 = affine.load %arg10[%arg19, %arg20, 256] : memref<257x257x257xf64>
        %1 = affine.load %arg10[%arg19, %arg20 + 1, 256] : memref<257x257x257xf64>
        %2 = arith.subf %0, %1 : f64
        %3 = affine.load %arg6[%arg19, %arg20] : memref<257x257xf64>
        %4 = arith.addf %2, %3 : f64
        %5 = affine.load %arg11[%arg19, %arg20, 256] : memref<257x257x257xf64>
        %6 = arith.subf %4, %5 : f64
        affine.store %6, %arg7[%arg19, %arg20] : memref<257x257xf64>
        %7 = affine.load %arg17[%arg20] : memref<257xf64>
        %8 = affine.load %arg18[%arg20] : memref<257xf64>
        %9 = arith.divf %7, %8 : f64
        %10 = affine.load %arg9[%arg19, %arg20, 256] : memref<257x257x257xf64>
        %11 = arith.mulf %9, %10 : f64
        %12 = arith.divf %arg4, %8 : f64
        %13 = arith.mulf %12, %6 : f64
        %14 = arith.subf %11, %13 : f64
        affine.store %14, %arg8[%arg19, %arg20] : memref<257x257xf64>
        %15 = affine.load %arg15[256] : memref<257xf64>
        %16 = affine.load %arg16[256] : memref<257xf64>
        %17 = arith.divf %15, %16 : f64
        %18 = affine.load %arg12[%arg19, %arg20, 256] : memref<257x257x257xf64>
        %19 = arith.mulf %17, %18 : f64
        %20 = affine.load %arg14[%arg19] : memref<257xf64>
        %21 = arith.mulf %arg3, %20 : f64
        %22 = arith.divf %21, %16 : f64
        %23 = arith.mulf %22, %14 : f64
        %24 = arith.addf %19, %23 : f64
        %25 = affine.load %arg13[%arg19] : memref<257xf64>
        %26 = arith.mulf %arg3, %25 : f64
        %27 = arith.divf %26, %16 : f64
        %28 = affine.load %arg9[%arg19, %arg20, 256] : memref<257x257x257xf64>
        %29 = arith.mulf %27, %28 : f64
        %30 = arith.subf %24, %29 : f64
        affine.store %30, %arg12[%arg19, %arg20, 256] : memref<257x257x257xf64>
        %31 = affine.load %arg8[%arg19, %arg20] : memref<257x257xf64>
        affine.store %31, %arg9[%arg19, %arg20, 256] : memref<257x257x257xf64>
        affine.for %arg21 = 0 to 256 {
          %64 = affine.load %arg10[%arg19, 256, %arg21] : memref<257x257x257xf64>
          %65 = affine.load %arg5[%arg19, %arg21] : memref<257x257xf64>
          %66 = arith.subf %64, %65 : f64
          %67 = affine.load %arg11[%arg19, 256, %arg21 + 1] : memref<257x257x257xf64>
          %68 = arith.addf %66, %67 : f64
          %69 = affine.load %arg11[%arg19, 256, %arg21] : memref<257x257x257xf64>
          %70 = arith.subf %68, %69 : f64
          affine.store %70, %arg7[%arg19, %arg20] : memref<257x257xf64>
          %71 = affine.load %arg17[256] : memref<257xf64>
          %72 = affine.load %arg18[%arg20] : memref<257xf64>
          %73 = arith.divf %71, %72 : f64
          %74 = affine.load %arg9[%arg19, %arg20, %arg21] : memref<257x257x257xf64>
          %75 = arith.mulf %73, %74 : f64
          %76 = arith.divf %arg4, %72 : f64
          %77 = arith.mulf %76, %70 : f64
          %78 = arith.subf %75, %77 : f64
          affine.store %78, %arg8[%arg19, %arg20] : memref<257x257xf64>
          %79 = affine.load %arg15[%arg21] : memref<257xf64>
          %80 = affine.load %arg16[%arg21] : memref<257xf64>
          %81 = arith.divf %79, %80 : f64
          %82 = affine.load %arg12[%arg19, 256, %arg21] : memref<257x257x257xf64>
          %83 = arith.mulf %81, %82 : f64
          %84 = affine.load %arg14[%arg19] : memref<257xf64>
          %85 = arith.mulf %arg3, %84 : f64
          %86 = arith.divf %85, %80 : f64
          %87 = arith.mulf %86, %78 : f64
          %88 = arith.addf %83, %87 : f64
          %89 = affine.load %arg13[%arg19] : memref<257xf64>
          %90 = arith.mulf %arg3, %89 : f64
          %91 = arith.divf %90, %80 : f64
          %92 = affine.load %arg9[%arg19, 256, %arg21] : memref<257x257x257xf64>
          %93 = arith.mulf %91, %92 : f64
          %94 = arith.subf %88, %93 : f64
          affine.store %94, %arg12[%arg19, 256, %arg21] : memref<257x257x257xf64>
          %95 = affine.load %arg8[%arg19, %arg20] : memref<257x257xf64>
          affine.store %95, %arg9[%arg19, 256, %arg21] : memref<257x257x257xf64>
        }
        %32 = affine.load %arg10[%arg19, 256, 256] : memref<257x257x257xf64>
        %33 = affine.load %arg5[%arg19, 256] : memref<257x257xf64>
        %34 = arith.subf %32, %33 : f64
        %35 = affine.load %arg6[%arg19, 256] : memref<257x257xf64>
        %36 = arith.addf %34, %35 : f64
        %37 = affine.load %arg11[%arg19, 256, 256] : memref<257x257x257xf64>
        %38 = arith.subf %36, %37 : f64
        affine.store %38, %arg7[%arg19, %arg20] : memref<257x257xf64>
        %39 = affine.load %arg17[256] : memref<257xf64>
        %40 = affine.load %arg18[256] : memref<257xf64>
        %41 = arith.divf %39, %40 : f64
        %42 = affine.load %arg9[%arg19, 256, 256] : memref<257x257x257xf64>
        %43 = arith.mulf %41, %42 : f64
        %44 = arith.divf %arg4, %40 : f64
        %45 = arith.mulf %44, %38 : f64
        %46 = arith.subf %43, %45 : f64
        affine.store %46, %arg8[%arg19, %arg20] : memref<257x257xf64>
        %47 = affine.load %arg15[256] : memref<257xf64>
        %48 = affine.load %arg16[256] : memref<257xf64>
        %49 = arith.divf %47, %48 : f64
        %50 = affine.load %arg12[%arg19, 256, 256] : memref<257x257x257xf64>
        %51 = arith.mulf %49, %50 : f64
        %52 = affine.load %arg14[%arg19] : memref<257xf64>
        %53 = arith.mulf %arg3, %52 : f64
        %54 = arith.divf %53, %48 : f64
        %55 = arith.mulf %54, %46 : f64
        %56 = arith.addf %51, %55 : f64
        %57 = affine.load %arg13[%arg19] : memref<257xf64>
        %58 = arith.mulf %arg3, %57 : f64
        %59 = arith.divf %58, %48 : f64
        %60 = affine.load %arg9[%arg19, 256, 256] : memref<257x257x257xf64>
        %61 = arith.mulf %59, %60 : f64
        %62 = arith.subf %56, %61 : f64
        affine.store %62, %arg12[%arg19, 256, 256] : memref<257x257x257xf64>
        %63 = affine.load %arg8[%arg19, %arg20] : memref<257x257xf64>
        affine.store %63, %arg9[%arg19, 256, 256] : memref<257x257x257xf64>
      }
    }
    return
  }
}

