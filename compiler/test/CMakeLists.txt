find_program(FILECHECK_EXECUTABLE FileCheck HINTS "${LLVM_TOOLS_BINARY_DIR}")

if(FILECHECK_EXECUTABLE)
  add_test(
    NAME check-treatment-effect-lowering
    COMMAND ${CMAKE_COMMAND}
      -DCI_OPT=$<TARGET_FILE:ci-opt>
      -DFILECHECK=${FILECHECK_EXECUTABLE}
      -DINPUT_MLIR=${CMAKE_CURRENT_SOURCE_DIR}/treatment-effect-lowering.mlir
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check-treatment-effect-lowering.cmake
  )
endif()
