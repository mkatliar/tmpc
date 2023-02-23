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
#include "ModelDimensions.hpp"

#include <benchmark/benchmark.h>


static void BM_casadiGeneratedImplicitDae(::benchmark::State& state)
{
    CasadiImplicitDae dae;
    blaze::StaticVector<double, NX> x, xdot;
    blaze::StaticVector<double, NZ> z;
    blaze::StaticVector<double, NU> u;
    blaze::StaticVector<double, NX + NZ> f;
    blaze::StaticMatrix<double, NX + NZ, NX, blaze::columnMajor> Jx, Jxdot;
    blaze::StaticMatrix<double, NX + NZ, NZ, blaze::columnMajor> Jz;

    for (auto _ : state)
        dae(0., xdot, x, z, u, f, Jxdot, Jx, Jz);
}

BENCHMARK(BM_casadiGeneratedImplicitDae);

