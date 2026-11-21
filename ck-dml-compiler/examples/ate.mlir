module {
  func.func public @query(%X: tensor<?x3xf64>, %b0: tensor<4xf64>, %b1: tensor<4xf64>) -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) {
    %a0_0 = "dml.ate"(%X, %b0) {delta = 1.00000000000000000e+00 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<f64>
    %a0_1 = "dml.ate"(%X, %b0) {delta = 2.00000000000000000e+00 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<f64>
    %a1_0 = "dml.ate"(%X, %b1) {delta = 1.00000000000000000e+00 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<f64>
    %a1_1 = "dml.ate"(%X, %b1) {delta = 2.00000000000000000e+00 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<f64>
    return %a0_0, %a0_1, %a1_0, %a1_1 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
  }
}
