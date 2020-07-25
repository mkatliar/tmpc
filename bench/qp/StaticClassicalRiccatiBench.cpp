#include <tmpc/qp/StaticOcpQp.hpp>
#include <tmpc/ocp/StaticOcpSolution.hpp>
#include <tmpc/qp/StaticClassicalRiccati.hpp>
#include <tmpc/qp/Randomize.hpp>

#include <bench/Complexity.hpp>


// #define BENCHMARK_STATIC_CLASSICAL_RICCATI(NX) \
//     BENCHMARK_TEMPLATE(BM_StaticClassicalRiccati, NX, 1)->Arg(100); \
//     BENCHMARK_TEMPLATE(BM_StaticClassicalRiccati, NX, NX)->Arg(100);

#define BENCHMARK_STATIC_CLASSICAL_RICCATI(NX) \
    BENCHMARK_TEMPLATE(BM_StaticClassicalRiccati, NX, NX)->Arg(100);


namespace tmpc :: benchmark
{
    template <size_t NX, size_t NU>
    void BM_StaticClassicalRiccati(State& state)
    {
        size_t const N = state.range(0);

        OcpTree const g(N + 1);
        StaticOcpQp<double, NX, NU> qp(g);
        StaticOcpSolution<double, NX, NU> sol(g);
        StaticClassicalRiccati<double, NX, NU> riccati(g);

        randomize(qp);

        for (auto _ : state)
            riccati(qp, sol);

        setCounters(state.counters, complexityClassicalRiccati(NX, NU, N));
        state.counters["nx"] = NX;
        state.counters["nu"] = NU;
        state.counters["n"] = N;
    }


    BENCHMARK_STATIC_CLASSICAL_RICCATI(1);

    BENCHMARK_STATIC_CLASSICAL_RICCATI(2);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(3);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(4);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(5);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(6);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(7);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(8);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(9);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(10);
    
    BENCHMARK_STATIC_CLASSICAL_RICCATI(11);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(12);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(13);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(14);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(15);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(16);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(17);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(18);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(19);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(20);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(21);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(22);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(23);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(24);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(25);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(26);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(27);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(28);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(29);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(30);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(31);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(32);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(33);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(34);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(35);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(36);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(37);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(38);
    BENCHMARK_STATIC_CLASSICAL_RICCATI(39);

    // BENCHMARK_STATIC_CLASSICAL_RICCATI(40);
    // BENCHMARK_STATIC_CLASSICAL_RICCATI(41);
    // BENCHMARK_STATIC_CLASSICAL_RICCATI(42);
    // BENCHMARK_STATIC_CLASSICAL_RICCATI(43);
    // BENCHMARK_STATIC_CLASSICAL_RICCATI(44);
    // BENCHMARK_STATIC_CLASSICAL_RICCATI(45);
    // BENCHMARK_STATIC_CLASSICAL_RICCATI(46);
    // BENCHMARK_STATIC_CLASSICAL_RICCATI(47);
    // BENCHMARK_STATIC_CLASSICAL_RICCATI(48);
    // BENCHMARK_STATIC_CLASSICAL_RICCATI(49);
    // BENCHMARK_STATIC_CLASSICAL_RICCATI(50);
}