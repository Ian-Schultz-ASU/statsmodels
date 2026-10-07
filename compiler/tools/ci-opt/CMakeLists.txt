add_llvm_executable(ci-opt
  ci-opt.cpp
)

llvm_update_compile_flags(ci-opt)

target_link_libraries(ci-opt
  PRIVATE
  CausalDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMlirOptMain
)

mlir_check_all_link_libraries(ci-opt)
