// After the full pipeline only standard dialects remain.
// RUN: dml-opt --dml-decompose --cse --dml-lower-to-linalg %s | FileCheck %s

func.func @interval(%X: tensor<?x3xf64>, %beta: tensor<4xf64>, %cov: tensor<4x4xf64>)
    -> (tensor<?xf64>, tensor<?xf64>) {
  %lo, %hi = "dml.effect_interval"(%X, %beta, %cov) {delta = 1.0 : f64, z = 2.0 : f64}
      : (tensor<?x3xf64>, tensor<4xf64>, tensor<4x4xf64>) -> (tensor<?xf64>, tensor<?xf64>)
  return %lo, %hi : tensor<?xf64>, tensor<?xf64>
}

// CHECK-LABEL: func.func @interval
// CHECK-NOT:   dml.
// The intercept is never materialized as a column of ones: no concat, no pad.
// CHECK-NOT:   tensor.concat
// CHECK-NOT:   tensor.pad

// cate: w = beta[1:], init = fill(beta[0]), theta = init + X @ w
// CHECK:       %[[W:.*]] = tensor.extract_slice %arg1[1] [3] [1]
// CHECK:       %[[B0:.*]] = tensor.extract %arg1[%{{.*}}]
// CHECK:       %[[INIT:.*]] = linalg.fill ins(%[[B0]] : f64)
// CHECK:       linalg.matvec ins(%arg0, %[[W]] : tensor<?x3xf64>, tensor<3xf64>) outs(%[[INIT]]

// cate_stderr: slices of cov, X @ S, one reduction, then sqrt
// CHECK:       tensor.extract_slice %arg2[0, 1] [1, 3] [1, 1]
// CHECK:       tensor.extract_slice %arg2[1, 1] [3, 3] [1, 1]
// CHECK:       linalg.matmul
// CHECK:       iterator_types = ["parallel", "reduction"]
// CHECK:       math.sqrt

// ---------------------------------------------------------------------------

func.func @ate(%X: tensor<?x3xf64>, %beta: tensor<4xf64>) -> tensor<f64> {
  %a = "dml.ate"(%X, %beta) {delta = 2.0 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<f64>
  return %a : tensor<f64>
}

// CHECK-LABEL: func.func @ate
// CHECK-NOT:   dml.
// The only loop over the rows of X is the column sum; no matvec is needed.
// CHECK-NOT:   linalg.matvec
// CHECK:       iterator_types = ["reduction", "parallel"]
// CHECK:       arith.divf
// CHECK:       linalg.dot
