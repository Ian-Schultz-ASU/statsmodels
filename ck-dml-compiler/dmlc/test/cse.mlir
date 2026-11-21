// Three scenarios on the same X and beta share one dml.cate after stock CSE.
// RUN: dml-opt --dml-decompose --cse %s | FileCheck %s
// Without CSE the three copies remain.
// RUN: dml-opt --dml-decompose %s | FileCheck %s --check-prefix=NOCSE

func.func @three_scenarios(%X: tensor<?x3xf64>, %beta: tensor<4xf64>)
    -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) {
  %e0 = "dml.effect"(%X, %beta) {delta = 0.5 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
  %e1 = "dml.effect"(%X, %beta) {delta = 1.0 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
  %e2 = "dml.effect"(%X, %beta) {delta = 2.0 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
  return %e0, %e1, %e2 : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>
}

// CHECK-LABEL:   func.func @three_scenarios
// CHECK-COUNT-1: "dml.cate"
// CHECK-NOT:     "dml.cate"
// CHECK-COUNT-3: "dml.scale"
// CHECK-NOT:     "dml.scale"

// NOCSE-LABEL:   func.func @three_scenarios
// NOCSE-COUNT-3: "dml.cate"
// NOCSE-NOT:     "dml.cate"
