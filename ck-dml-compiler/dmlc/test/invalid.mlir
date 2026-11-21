// One case per verifier rule. --split-input-file checks each chunk on its own.
// RUN: not dml-opt --split-input-file %s 2>&1 | FileCheck %s

// CHECK: 'dml.effect' op beta must have one entry per feature plus an intercept: len(beta) == X.shape[1] + 1 (got 7 and 3)
func.func @bad_beta_len(%X: tensor<?x3xf64>, %b: tensor<7xf64>) -> tensor<?xf64> {
  %e = "dml.effect"(%X, %b) {delta = 1.0 : f64} : (tensor<?x3xf64>, tensor<7xf64>) -> tensor<?xf64>
  return %e : tensor<?xf64>
}

// -----
// CHECK: 'dml.cate' op X must be a rank-2 tensor, got rank 1
func.func @bad_x_rank(%X: tensor<?xf64>, %b: tensor<4xf64>) -> tensor<?xf64> {
  %t = "dml.cate"(%X, %b) : (tensor<?xf64>, tensor<4xf64>) -> tensor<?xf64>
  return %t : tensor<?xf64>
}

// -----
// CHECK: 'dml.cate' op theta must be a rank-1 tensor, got rank 2
func.func @bad_result_rank(%X: tensor<?x3xf64>, %b: tensor<4xf64>) -> tensor<?x9xf64> {
  %t = "dml.cate"(%X, %b) : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?x9xf64>
  return %t : tensor<?x9xf64>
}

// -----
// CHECK: 'dml.cate' op per-row result must have one entry per input row (got 5 and 8)
func.func @bad_rows(%X: tensor<8x3xf64>, %b: tensor<4xf64>) -> tensor<5xf64> {
  %t = "dml.cate"(%X, %b) : (tensor<8x3xf64>, tensor<4xf64>) -> tensor<5xf64>
  return %t : tensor<5xf64>
}

// -----
// CHECK: 'dml.cate_stderr' op cov must be (p, p) with p == X.shape[1] + 1 (got 3 and 3)
func.func @bad_cov(%X: tensor<?x3xf64>, %c: tensor<3x3xf64>) -> tensor<?xf64> {
  %s = "dml.cate_stderr"(%X, %c) : (tensor<?x3xf64>, tensor<3x3xf64>) -> tensor<?xf64>
  return %s : tensor<?xf64>
}

// -----
// CHECK: 'dml.effect_interval' op z must be positive
func.func @bad_z(%X: tensor<?x3xf64>, %b: tensor<4xf64>, %c: tensor<4x4xf64>) -> tensor<?xf64> {
  %lo, %hi = "dml.effect_interval"(%X, %b, %c) {delta = 1.0 : f64, z = 0.0 : f64}
      : (tensor<?x3xf64>, tensor<4xf64>, tensor<4x4xf64>) -> (tensor<?xf64>, tensor<?xf64>)
  return %lo : tensor<?xf64>
}

// -----
// CHECK: 'dml.ate' op ate must be a rank-0 tensor, got rank 1
func.func @bad_ate_rank(%X: tensor<?x3xf64>, %b: tensor<4xf64>) -> tensor<?xf64> {
  %a = "dml.ate"(%X, %b) {delta = 1.0 : f64} : (tensor<?x3xf64>, tensor<4xf64>) -> tensor<?xf64>
  return %a : tensor<?xf64>
}

// -----
// CHECK: 'dml.axpby' op x and y must have the same length (got 4 and 5)
func.func @bad_axpby(%x: tensor<4xf64>, %y: tensor<5xf64>) -> tensor<4xf64> {
  %o = "dml.axpby"(%x, %y) {a = 1.0 : f64, b = 1.0 : f64} : (tensor<4xf64>, tensor<5xf64>) -> tensor<4xf64>
  return %o : tensor<4xf64>
}

// -----
// CHECK: 'dml.col_mean' op xbar must have one entry per feature: len(xbar) == X.shape[1] (got 2 and 3)
func.func @bad_xbar(%X: tensor<?x3xf64>) -> tensor<2xf64> {
  %m = "dml.col_mean"(%X) : (tensor<?x3xf64>) -> tensor<2xf64>
  return %m : tensor<2xf64>
}

// -----
// CHECK: 'dml.ate_from_mean' op beta must have one entry per feature plus an intercept: len(beta) == len(xbar) + 1 (got 3 and 3)
func.func @bad_ate_from_mean(%m: tensor<3xf64>, %b: tensor<3xf64>) -> tensor<f64> {
  %a = "dml.ate_from_mean"(%m, %b) {delta = 1.0 : f64} : (tensor<3xf64>, tensor<3xf64>) -> tensor<f64>
  return %a : tensor<f64>
}

// -----
// CHECK: 'dml.mean_scale' op out must be a rank-0 tensor, got rank 1
func.func @bad_mean_scale(%x: tensor<4xf64>) -> tensor<1xf64> {
  %o = "dml.mean_scale"(%x) {a = 1.0 : f64} : (tensor<4xf64>) -> tensor<1xf64>
  return %o : tensor<1xf64>
}

// -----
// Element type is enforced by the op definition (ranked tensor of f64).
// CHECK: 'dml.scale' op operand #0 must be ranked tensor of 64-bit float values
func.func @bad_element_type(%x: tensor<4xf32>) -> tensor<4xf32> {
  %o = "dml.scale"(%x) {a = 1.0 : f64} : (tensor<4xf32>) -> tensor<4xf32>
  return %o : tensor<4xf32>
}
