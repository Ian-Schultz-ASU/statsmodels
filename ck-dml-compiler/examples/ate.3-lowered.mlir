#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0) -> (d0)>
module {
  func.func public @query(%arg0: tensor<?x3xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) {
    %0 = tensor.empty() : tensor<3xf64>
    %cst = arith.constant 0.000000e+00 : f64
    %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<3xf64>) -> tensor<3xf64>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<?x3xf64>) outs(%1 : tensor<3xf64>) {
    ^bb0(%in: f64, %out: f64):
      %23 = arith.addf %out, %in : f64
      linalg.yield %23 : f64
    } -> tensor<3xf64>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x3xf64>
    %3 = arith.index_cast %dim : index to i64
    %4 = arith.sitofp %3 : i64 to f64
    %5 = tensor.empty() : tensor<3xf64>
    %6 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%2 : tensor<3xf64>) outs(%5 : tensor<3xf64>) {
    ^bb0(%in: f64, %out: f64):
      %23 = arith.divf %in, %4 : f64
      linalg.yield %23 : f64
    } -> tensor<3xf64>
    %extracted_slice = tensor.extract_slice %arg1[1] [3] [1] : tensor<4xf64> to tensor<3xf64>
    %c0_0 = arith.constant 0 : index
    %extracted = tensor.extract %arg1[%c0_0] : tensor<4xf64>
    %7 = tensor.empty() : tensor<f64>
    %8 = linalg.fill ins(%extracted : f64) outs(%7 : tensor<f64>) -> tensor<f64>
    %9 = linalg.dot ins(%6, %extracted_slice : tensor<3xf64>, tensor<3xf64>) outs(%8 : tensor<f64>) -> tensor<f64>
    %extracted_1 = tensor.extract %9[] : tensor<f64>
    %cst_2 = arith.constant 1.000000e+00 : f64
    %10 = arith.mulf %extracted_1, %cst_2 : f64
    %from_elements = tensor.from_elements %10 : tensor<f64>
    %extracted_slice_3 = tensor.extract_slice %arg1[1] [3] [1] : tensor<4xf64> to tensor<3xf64>
    %c0_4 = arith.constant 0 : index
    %extracted_5 = tensor.extract %arg1[%c0_4] : tensor<4xf64>
    %11 = tensor.empty() : tensor<f64>
    %12 = linalg.fill ins(%extracted_5 : f64) outs(%11 : tensor<f64>) -> tensor<f64>
    %13 = linalg.dot ins(%6, %extracted_slice_3 : tensor<3xf64>, tensor<3xf64>) outs(%12 : tensor<f64>) -> tensor<f64>
    %extracted_6 = tensor.extract %13[] : tensor<f64>
    %cst_7 = arith.constant 2.000000e+00 : f64
    %14 = arith.mulf %extracted_6, %cst_7 : f64
    %from_elements_8 = tensor.from_elements %14 : tensor<f64>
    %extracted_slice_9 = tensor.extract_slice %arg2[1] [3] [1] : tensor<4xf64> to tensor<3xf64>
    %c0_10 = arith.constant 0 : index
    %extracted_11 = tensor.extract %arg2[%c0_10] : tensor<4xf64>
    %15 = tensor.empty() : tensor<f64>
    %16 = linalg.fill ins(%extracted_11 : f64) outs(%15 : tensor<f64>) -> tensor<f64>
    %17 = linalg.dot ins(%6, %extracted_slice_9 : tensor<3xf64>, tensor<3xf64>) outs(%16 : tensor<f64>) -> tensor<f64>
    %extracted_12 = tensor.extract %17[] : tensor<f64>
    %cst_13 = arith.constant 1.000000e+00 : f64
    %18 = arith.mulf %extracted_12, %cst_13 : f64
    %from_elements_14 = tensor.from_elements %18 : tensor<f64>
    %extracted_slice_15 = tensor.extract_slice %arg2[1] [3] [1] : tensor<4xf64> to tensor<3xf64>
    %c0_16 = arith.constant 0 : index
    %extracted_17 = tensor.extract %arg2[%c0_16] : tensor<4xf64>
    %19 = tensor.empty() : tensor<f64>
    %20 = linalg.fill ins(%extracted_17 : f64) outs(%19 : tensor<f64>) -> tensor<f64>
    %21 = linalg.dot ins(%6, %extracted_slice_15 : tensor<3xf64>, tensor<3xf64>) outs(%20 : tensor<f64>) -> tensor<f64>
    %extracted_18 = tensor.extract %21[] : tensor<f64>
    %cst_19 = arith.constant 2.000000e+00 : f64
    %22 = arith.mulf %extracted_18, %cst_19 : f64
    %from_elements_20 = tensor.from_elements %22 : tensor<f64>
    return %from_elements, %from_elements_8, %from_elements_14, %from_elements_20 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
  }
}

