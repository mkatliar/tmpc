tmpc_generate_casadi_function(GENERATED_SRCS GENERATED_HDRS "generated_gemm.py")

add_executable(bench-tmpc-casadi
    Main.cpp
    Gemm.cpp
    ${GENERATED_SRCS} 
    ${GENERATED_HDRS}
)
    
target_include_directories(bench-tmpc-casadi
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(bench-tmpc-casadi
    tmpc-casadi
    benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)
