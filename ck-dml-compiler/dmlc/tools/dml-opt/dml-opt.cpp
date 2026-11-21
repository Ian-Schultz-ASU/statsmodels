#include "DML/DML.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerTransformsPasses(); // stock passes, including --cse
  dml::registerDMLPasses();

  mlir::DialectRegistry registry;
  registry.insert<dml::DMLDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::tensor::TensorDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Double ML inference optimizer\n", registry));
}
