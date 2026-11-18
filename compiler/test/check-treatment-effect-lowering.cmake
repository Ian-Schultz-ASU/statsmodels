execute_process(
  COMMAND "${CI_OPT}" --ci-lower-treatment-effect "${INPUT_MLIR}"
  OUTPUT_VARIABLE lowered_mlir
  RESULT_VARIABLE opt_status
)
if(NOT opt_status EQUAL 0)
  message(FATAL_ERROR "ci-opt failed with exit code ${opt_status}")
endif()

execute_process(
  COMMAND "${FILECHECK}" "${INPUT_MLIR}"
  INPUT "${lowered_mlir}"
  RESULT_VARIABLE check_status
)
if(NOT check_status EQUAL 0)
  message(FATAL_ERROR "FileCheck failed with exit code ${check_status}")
endif()
