#include <tmpc/blasfeo/Blasfeo.hpp>

#include <benchmark/benchmark.h>

#include <random>
#include <memory>


namespace tmpc :: benchmark
{
    template <typename MT>
    static void randomize(blasfeo::Matrix<MT>& A)
    {
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> dis(-1.0, 1.0);	

        for (size_t i = 0; i < rows(~A); ++i)
            for (size_t j = 0; j < columns(~A); ++j)
                (~A)(i, j) = dis(gen);
    }


    static void BM_gemm(::benchmark::State& state)
    {
        size_t const m = state.range(0);
        size_t const n = state.range(1);
        size_t const k = state.range(2);

        blasfeo::DynamicMatrix<double> A(k, m), B(k, n), C(m, n), D(m, n);

        randomize(A);
        randomize(B);
        randomize(C);

        /// @brief D <= beta * C + alpha * A^T * B
        // inline void gemm_tn(size_t m, size_t n, size_t k,
        //     double alpha,
        //     blasfeo_dmat const& sA, size_t ai, size_t aj,
        //     blasfeo_dmat const& sB, size_t bi, size_t bj,
        //     double beta,
        //     blasfeo_dmat const& sC, size_t ci, size_t cj,
        //     blasfeo_dmat& sD, size_t di, size_t dj);
        
        for (auto _ : state)
            gemm_tn(m, n, k, 1., A, 0, 0, B, 0, 0, 1., C, 0, 0, D, 0, 0);
    }
    

    BENCHMARK(BM_gemm)
        ->Args({2, 2, 2})
        ->Args({3, 3, 3})
        ->Args({5, 5, 5})
        ->Args({10, 10, 10})
        ->Args({20, 20, 20})
        ->Args({30, 30, 30});
}