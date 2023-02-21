find_package(BLAS)

# Generate C code from CasADi functions
tmpc_generate_casadi_function(SRCS HDRS "pendulum_model.py")

add_executable(example-pendulum_on_cart-simulate
    Simulate.cpp
    "${SRCS}" "${HDRS}"
)

# Generated headers for CasADi functions is emitted to "${CMAKE_CURRENT_BINARY_DIR}",
# and we need to add it to the include path s.t. the compiler can find them.
target_include_directories(example-pendulum_on_cart-simulate
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
)

target_link_libraries(example-pendulum_on_cart-simulate
    tmpc-casadi
    BLAS::BLAS
)
