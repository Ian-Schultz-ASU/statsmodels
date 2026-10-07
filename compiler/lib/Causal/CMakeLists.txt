add_mlir_dialect_library(CausalDialect
  CausalDialect.cpp

  DEPENDS
  CausalDialectIncGen
  CausalOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

target_include_directories(CausalDialect
  PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}/include"
)
