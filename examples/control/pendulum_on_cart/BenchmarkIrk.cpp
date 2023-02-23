// Copyright 2023 Mikhail Katliar
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "CasadiImplicitDae.hpp"
#include "CasadiImplicitDaeS.hpp"
#include "ModelDimensions.hpp"

#include <tmpc/integrator/StaticImplicitRungeKutta.hpp>
#include <tmpc/integrator/GaussRadauIIAMethod.hpp>

#include <benchmark/benchmark.h>


static void BM_irkWithSensitivities(::benchmark::State& state)
{
    CasadiImplicitDae dae;
    CasadiImplicitDaeS dae_s;
    tmpc::StaticImplicitRungeKutta<double, 3, NX, NZ, NU> irk {tmpc::GaussRadauIIAMethod {3}};

    blaze::StaticVector<double, NX> const x0 {5.863919e-02, 3.176782e+00, -2.557677e-01, 3.518985e+00};
    blaze::StaticMatrix<double, NX, NX + NU> Sx(0.);
    diagonal(Sx) = 1.;
    blaze::StaticVector<double, NU> u(0.);
    blaze::StaticVector<double, NX> xf;
    blaze::StaticMatrix<double, NX, NX + NU> Sxf;

    for (auto _ : state)
        tmpc::integrate(irk, dae, dae_s, 0., 0.01, 3, x0, Sx, u, xf, Sxf);
}

BENCHMARK(BM_irkWithSensitivities);

