// Average effect for 2 models x 2 scenarios on one population.
// With the linearity rewrite the only pass over X is a single col_mean.
// RUN: dml-opt --dml-decompose --cse %s | FileCheck %s --check-prefix=PUSH
// Without it, each model still needs its own pass over X (one cate per model).
// RUN: dml-opt --dml-decompose-no-pushdown --cse %s | FileCheck %s --check-prefix=NOPUSH
// RUN: dml-opt --dml-decompose-no-pushdown --cse %s | FileCheck %s --check-prefix=NOPUSH-MEAN

func.func @ate(%X: tensor<?x3xf64>, %b0: tensor<4xf64>, %b1: tensor<4xf64>)
    -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) {
  %a0 = "dml.ate"(%X, %b0) {delta = 1.0 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<f64>
  %a1 = "dml.ate"(%X, %b0) {delta = 2.0 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<f64>
  %a2 = "dml.ate"(%X, %b1) {delta = 1.0 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<f64>
  %a3 = "dml.ate"(%X, %b1) {delta = 2.0 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<f64>
  return %a0, %a1, %a2, %a3 : tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>
}

// PUSH-LABEL:   func.func @ate
// PUSH-NOT:     "dml.cate"
// PUSH-COUNT-1: "dml.col_mean"
// PUSH-NOT:     "dml.col_mean"
// PUSH-COUNT-4: "dml.ate_from_mean"
// PUSH-NOT:     "dml.ate_from_mean"

// NOPUSH-LABEL:   func.func @ate
// NOPUSH-NOT:     "dml.col_mean"
// NOPUSH-COUNT-2: "dml.cate"
// NOPUSH-NOT:     "dml.cate"

// NOPUSH-MEAN-LABEL:   func.func @ate
// NOPUSH-MEAN-COUNT-4: "dml.mean_scale"
// NOPUSH-MEAN-NOT:     "dml.mean_scale"
