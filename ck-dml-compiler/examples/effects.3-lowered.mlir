#map = affine_map<(d0) -> (d0)>
module {
  func.func public @query(%arg0: tensor<?x3xf64>, %arg1: tensor<4xf64>) -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) {
    %extracted_slice = tensor.extract_slice %arg1[1] [3] [1] : tensor<4xf64> to tensor<3xf64>
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg1[%c0] : tensor<4xf64>
    %c0_0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0_0 : tensor<?x3xf64>
    %0 = tensor.empty(%dim) : tensor<?xf64>
    %1 = linalg.fill ins(%extracted : f64) outs(%0 : tensor<?xf64>) -> tensor<?xf64>
    %2 = linalg.matvec ins(%arg0, %extracted_slice : tensor<?x3xf64>, tensor<3xf64>) outs(%1 : tensor<?xf64>) -> tensor<?xf64>
    %c0_1 = arith.constant 0 : index
    %dim_2 = tensor.dim %2, %c0_1 : tensor<?xf64>
    %3 = tensor.empty(%dim_2) : tensor<?xf64>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%2 : tensor<?xf64>) outs(%3 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %cst = arith.constant 5.000000e-01 : f64
      %9 = arith.mulf %in, %cst : f64
      linalg.yield %9 : f64
    } -> tensor<?xf64>
    %c0_3 = arith.constant 0 : index
    %dim_4 = tensor.dim %2, %c0_3 : tensor<?xf64>
    %5 = tensor.empty(%dim_4) : tensor<?xf64>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%2 : tensor<?xf64>) outs(%5 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %cst = arith.constant 1.000000e+00 : f64
      %9 = arith.mulf %in, %cst : f64
      linalg.yield %9 : f64
    } -> tensor<?xf64>
    %c0_5 = arith.constant 0 : index
    %dim_6 = tensor.dim %2, %c0_5 : tensor<?xf64>
    %7 = tensor.empty(%dim_6) : tensor<?xf64>
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%2 : tensor<?xf64>) outs(%7 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %cst = arith.constant 2.000000e+00 : f64
      %9 = arith.mulf %in, %cst : f64
      linalg.yield %9 : f64
    } -> tensor<?xf64>
    return %4, %6, %8 : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>
  }
}

