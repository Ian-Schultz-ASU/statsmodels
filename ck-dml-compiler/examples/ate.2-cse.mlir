module {
  func.func public @query(%arg0: tensor<?x3xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) {
    %0 = "dml.col_mean"(%arg0) : (tensor<?x3xf64>) -> tensor<3xf64>
    %1 = "dml.ate_from_mean"(%0, %arg1) <{delta = 1.000000e+00 : f64}> : (tensor<3xf64>, tensor<4xf64>) -> tensor<f64>
    %2 = "dml.ate_from_mean"(%0, %arg1) <{delta = 2.000000e+00 : f64}> : (tensor<3xf64>, tensor<4xf64>) -> tensor<f64>
    %3 = "dml.ate_from_mean"(%0, %arg2) <{delta = 1.000000e+00 : f64}> : (tensor<3xf64>, tensor<4xf64>) -> tensor<f64>
    %4 = "dml.ate_from_mean"(%0, %arg2) <{delta = 2.000000e+00 : f64}> : (tensor<3xf64>, tensor<4xf64>) -> tensor<f64>
    return %1, %2, %3, %4 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
  }
}

