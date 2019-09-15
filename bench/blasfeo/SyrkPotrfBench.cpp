#include <tmpc/blasfeo/Blasfeo.hpp>

#include <blaze/Math.h>

#include <benchmark/benchmark.h>


namespace tmpc :: testing
{
    void BM_SyrkPotrf_blasfeo(::benchmark::State& state)
    {
        size_t const m = state.range(0), k = state.range(1);

        // Init Blaze matrices
        //
        blaze::DynamicMatrix<double, blaze::columnMajor> blaze_A(m, k), blaze_C(m, m), blaze_D(m, m);
        randomize(blaze_A);
        makePositiveDefinite(blaze_C);
        
        // Init BLASFEO matrices
        //
        blasfeo::DynamicMatrix<double> blasfeo_A(blaze_A), blasfeo_C(blaze_C), blasfeo_D(m, m);
        
        // Do syrk-potrf with BLASFEO
        for (auto _ : state)
            syrk_potrf(m, k, blasfeo_A, 0, 0, blasfeo_A, 0, 0, blasfeo_C, 0, 0, blasfeo_D, 0, 0);
    }


    BENCHMARK(BM_SyrkPotrf_blasfeo)->Args({5, 4})->Args({35, 30});
}