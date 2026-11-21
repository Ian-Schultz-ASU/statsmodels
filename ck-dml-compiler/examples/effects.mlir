module {
  func.func public @query(%X: tensor<?x3xf64>, %beta: tensor<4xf64>) -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) {
    %e0 = "dml.effect"(%X, %beta) {delta = 5.00000000000000000e-01 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
    %e1 = "dml.effect"(%X, %beta) {delta = 1.00000000000000000e+00 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
    %e2 = "dml.effect"(%X, %beta) {delta = 2.00000000000000000e+00 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
    return %e0, %e1, %e2 : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>
  }
}
