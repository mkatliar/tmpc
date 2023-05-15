find_package(benchmark REQUIRED)

#~~~~~~~~~~~~~~~~
add_executable(bench-tmpc
    Main.cpp

    math/Cholesky.cpp
    math/Trsv.cpp
    math/SyrkPotrf.cpp

    # CondensingBench.cpp
    qp/DynamicClassicalRiccatiBench.cpp
    qp/DynamicFactorizedRiccatiBench.cpp
    qp/StaticClassicalRiccatiBench.cpp
    qp/StaticFactorizedRiccatiBench.cpp
    OcpTreeBench.cpp
)


target_link_libraries(bench-tmpc
    tmpc
    benchmark::benchmark
    blast-blasfeo
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)


if (TMPC_WITH_CASADI)
    add_subdirectory("casadi")
endif ()


if (TMPC_WITH_HPIPM)
    add_subdirectory("qp-hpipm")
endif ()
