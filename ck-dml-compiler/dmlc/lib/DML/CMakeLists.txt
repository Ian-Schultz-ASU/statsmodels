add_mlir_dialect_library(DMLDialect
  DMLDialect.cpp
  DMLPasses.cpp
  DEPENDS
  MLIRDMLOpsIncGen
  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
)
