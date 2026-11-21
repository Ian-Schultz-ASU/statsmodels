// Lowering only understands building-block ops. A query op is a clear error.
// RUN: not dml-opt --dml-lower-to-linalg %s 2>&1 | FileCheck %s

func.func @q(%X: tensor<?x3xf64>, %beta: tensor<4xf64>) -> tensor<?xf64> {
  %e = "dml.effect"(%X, %beta) {delta = 1.0 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
  return %e : tensor<?xf64>
}

// CHECK: 'dml.effect' op run --dml-decompose before --dml-lower-to-linalg
