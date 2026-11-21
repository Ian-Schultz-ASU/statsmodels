// Each query op is rewritten into building-block ops; no query op survives.
// RUN: dml-opt --dml-decompose %s | FileCheck %s

func.func @queries(%X: tensor<?x3xf64>, %beta: tensor<4xf64>, %cov: tensor<4x4xf64>)
    -> (tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<f64>) {
  %e = "dml.effect"(%X, %beta) {delta = 2.0 : f64}
      : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
  %lo, %hi = "dml.effect_interval"(%X, %beta, %cov) {delta = -2.0 : f64, z = 1.5 : f64}
      : (tensor<?x3xf64>, tensor<4xf64>, tensor<4x4xf64>) -> (tensor<?xf64>, tensor<?xf64>)
  %a = "dml.ate"(%X, %beta) {delta = 2.0 : f64}
      : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<f64>
  return %e, %lo, %hi, %a : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>, tensor<f64>
}

// CHECK-LABEL: func.func @queries
// CHECK-NOT:   dml.effect
// CHECK-NOT:   "dml.ate"

// effect -> cate + scale by delta
// CHECK:       %[[T0:.*]] = "dml.cate"(%arg0, %arg1)
// CHECK:       "dml.scale"(%[[T0]]) <{a = 2.000000e+00 : f64}>

// effect_interval -> cate + cate_stderr + two axpby.
// The half width is z * |delta| = 1.5 * 2 = 3, also for a negative delta.
// CHECK:       %[[T1:.*]] = "dml.cate"(%arg0, %arg1)
// CHECK:       %[[SE:.*]] = "dml.cate_stderr"(%arg0, %arg2)
// CHECK:       "dml.axpby"(%[[T1]], %[[SE]]) <{a = -2.000000e+00 : f64, b = -3.000000e+00 : f64}>
// CHECK:       "dml.axpby"(%[[T1]], %[[SE]]) <{a = -2.000000e+00 : f64, b = 3.000000e+00 : f64}>

// ate -> col_mean + ate_from_mean (the linearity rewrite)
// CHECK:       %[[XB:.*]] = "dml.col_mean"(%arg0)
// CHECK:       "dml.ate_from_mean"(%[[XB]], %arg1) <{delta = 2.000000e+00 : f64}>
