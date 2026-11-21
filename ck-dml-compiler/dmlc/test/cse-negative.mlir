// CSE must not merge work that differs: two models (different beta) keep two
// dml.cate ops, and two scenarios with the same delta collapse to one scale.
// Each op kind is counted by its own FileCheck run, because the two kinds
// are interleaved in the output.
// RUN: dml-opt --dml-decompose --cse %s | FileCheck %s --check-prefix=CATE
// RUN: dml-opt --dml-decompose --cse %s | FileCheck %s --check-prefix=SCALE

func.func @two_models(%X: tensor<?x3xf64>, %b0: tensor<4xf64>, %b1: tensor<4xf64>)
    -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) {
  %e0 = "dml.effect"(%X, %b0) {delta = 1.0 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
  %e1 = "dml.effect"(%X, %b1) {delta = 1.0 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
  %e2 = "dml.effect"(%X, %b1) {delta = 1.0 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
  return %e0, %e1, %e2 : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>
}

// CATE-LABEL:   func.func @two_models
// CATE-COUNT-2: "dml.cate"
// CATE-NOT:     "dml.cate"

// SCALE-LABEL:   func.func @two_models
// SCALE-COUNT-2: "dml.scale"
// SCALE-NOT:     "dml.scale"
