find_package(BLAS)

#
# System model library
#
tmpc_generate_casadi_function(GENERATED_SRCS GENERATED_HDRS "pendulum_model.py")

add_library(example-pendulum_on_cart-model STATIC
    "${GENERATED_SRCS}"
    "${GENERATED_HDRS}"
    CasadiImplicitDae.cpp
    CasadiImplicitDaeS.cpp
)

target_link_libraries(example-pendulum_on_cart-model
    tmpc-casadi
)

add_executable(example-pendulum_on_cart-simulate
    Simulate.cpp
)

# Generated headers for CasADi functions is emitted to "${CMAKE_CURRENT_BINARY_DIR}",
# and we need to add it to the include path s.t. the compiler can find them.
target_include_directories(example-pendulum_on_cart-model
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
)

target_link_libraries(example-pendulum_on_cart-simulate
    example-pendulum_on_cart-model
    BLAS::BLAS
)

# if (TMPC_WITH_BENCHMARK)
    find_package(benchmark REQUIRED)

    add_executable(example-pendulum_on_cart-benchmark
        BenchmarkMain.cpp
        BenchmarkCasadiDae.cpp
    )

    target_link_libraries(example-pendulum_on_cart-benchmark
        example-pendulum_on_cart-model
        benchmark::benchmark
    )
# endif ()