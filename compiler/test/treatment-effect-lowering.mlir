// RUN: ci-opt --ci-lower-treatment-effect %s | FileCheck %s

module {
  func.func public @counterfactual(
      %data: tensor<4x2xf64>,
      %coefficients: tensor<5xf64>
  ) -> tensor<4xf64> {
    %effects = "ci.treatment_effect"(%data, %coefficients) {
      control_value = 0.0 : f64,
      interaction_columns = ["x1", "x2"],
      interaction_coefficients = array<i64: 3, 4>,
      main_effect_index = 1 : i64,
      treated_value = 2.0 : f64,
      treatment_name = "T"
    } : (tensor<4x2xf64>, tensor<5xf64>) -> tensor<4xf64>
    return %effects : tensor<4xf64>
  }
}

// CHECK-LABEL: func.func public @counterfactual
// CHECK-NOT: ci.treatment_effect
// CHECK: tensor.from_elements
// CHECK: linalg.fill
// CHECK: linalg.matvec
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: arith.mulf
