// The number of features must be known at compile time (the lowering slices
// beta and cov by it). The number of rows may be dynamic.
// RUN: not dml-opt --dml-lower-to-linalg %s 2>&1 | FileCheck %s

func.func @q(%X: tensor<?x?xf64>, %beta: tensor<?xf64>) -> tensor<?xf64> {
  %t = "dml.cate"(%X, %beta) : (tensor<?x?xf64>, tensor<?xf64>) -> tensor<?xf64>
  return %t : tensor<?xf64>
}

// CHECK: 'dml.cate' op feature count must be static
