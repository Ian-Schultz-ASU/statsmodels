module {
  func.func public @query(%arg0: tensor<?x3xf64>, %arg1: tensor<4xf64>) -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) {
    %0 = "dml.cate"(%arg0, %arg1) : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
    %1 = "dml.scale"(%0) <{a = 5.000000e-01 : f64}> : (tensor<?xf64>) -> tensor<?xf64>
    %2 = "dml.cate"(%arg0, %arg1) : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
    %3 = "dml.scale"(%2) <{a = 1.000000e+00 : f64}> : (tensor<?xf64>) -> tensor<?xf64>
    %4 = "dml.cate"(%arg0, %arg1) : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
    %5 = "dml.scale"(%4) <{a = 2.000000e+00 : f64}> : (tensor<?xf64>) -> tensor<?xf64>
    return %1, %3, %5 : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>
  }
}

