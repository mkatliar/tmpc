cmake_minimum_required (VERSION 3.10)   # Need at least 3.10 for gtest_discover_tests()
project (tmpc)

# Enable modern C++
set(CMAKE_CXX_STANDARD 20)

# Don't use compiler's C++ extensions
set(CMAKE_CXX_EXTENSIONS False)

# Enable position-independent code on all targets by default.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include CMakeToolsHelpers for easier debugging from VS Code
include(CMakeToolsHelpers OPTIONAL)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)
set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_DEBUG_POSTFIX d)

# List of QP solvers
set(qp_solvers "qpOASES" "HPMPC" "HPIPM" "treeQP")

# List of selectable extern libs
set(extern_libs ${qp_solvers})

# Create TMPC_WITH_${lib} cached variables and find selected packages
foreach (lib ${extern_libs})
    option(TMPC_WITH_${lib} "Link to ${lib}")

    if (TMPC_WITH_${lib})
        find_package(${lib} REQUIRED)
    endif ()
endforeach ()

find_package(blaze 3.8 REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

option(TMPC_WITH_CASADI "Build tmpc with CasADi generated function support classes")
option(TMPC_WITH_JSON "Build tmpc with JSON support")
option(TMPC_WITH_EXAMPLES "Build tmpc examples")
option(TMPC_WITH_TEST "Build tmpc tests")
option(TMPC_WITH_BENCHMARK "Build tmpc benchmarks")

add_subdirectory(blast)
add_subdirectory(src)

if (TMPC_WITH_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (TMPC_WITH_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

if (TMPC_WITH_BENCHMARK)
    add_subdirectory(bench)
endif()
