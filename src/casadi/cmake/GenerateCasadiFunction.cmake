function(tmpc_generate_casadi_function SRCS HDRS PYTHON_FILES)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  if(NOT PYTHON_FILES)
    message(SEND_ERROR "Error: tmpc_generate_casadi_function() called without any python files")
    return()
  endif()

  set(TMPC_GENERATE_CASADI_FUNCTION_SCRIPT "${PROJECT_SOURCE_DIR}/src/casadi/python/generate_casadi_function.py")

  foreach(FIL ${PYTHON_FILES})
    get_filename_component(ABS_FIL "${FIL}" ABSOLUTE)
    get_filename_component(FIL_WE "${FIL}" NAME_WE)
    get_filename_component(FIL_DIR "${FIL}" DIRECTORY)

    set(_casadi_c_src "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.c")
    set(_casadi_c_hdr "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.h")
    list(APPEND ${SRCS} "${_casadi_c_src}")
    list(APPEND ${HDRS} "${_casadi_c_hdr}")

    add_custom_command(
        OUTPUT
            "${_casadi_c_src}"
            "${_casadi_c_hdr}"
        COMMAND
            "${Python3_EXECUTABLE}"
        ARGS
            "${TMPC_GENERATE_CASADI_FUNCTION_SCRIPT}"
            "${ABS_FIL}"
        WORKING_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}"
        DEPENDS
            "${FIL}"
            "${TMPC_GENERATE_CASADI_FUNCTION_SCRIPT}"
        COMMENT
            "Generating code for casadi functions in ${FIL_WE}..."
        VERBATIM
    )
  endforeach()

  set(${SRCS} "${${SRCS}}" PARENT_SCOPE)
  set(${HDRS} "${${HDRS}}" PARENT_SCOPE)
endfunction()