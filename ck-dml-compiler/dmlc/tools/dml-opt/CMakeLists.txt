add_llvm_executable(dml-opt dml-opt.cpp)
llvm_update_compile_flags(dml-opt)
target_link_libraries(dml-opt PRIVATE DMLDialect MLIROptLib)
