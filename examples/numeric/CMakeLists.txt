add_executable(newton_solver newton_solver.cpp)
target_link_libraries(newton_solver 
    tmpc
    ${LAPACK_LIBRARIES}
)