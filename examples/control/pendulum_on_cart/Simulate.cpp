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

#include "ModelDimensions.hpp"
#include "CasadiImplicitDae.hpp"
#include "CasadiImplicitDaeS.hpp"

#include <tmpc/integrator/StaticImplicitRungeKutta.hpp>
#include <tmpc/integrator/GaussRadauIIAMethod.hpp>

#include <iostream>


int main(int, char **)
{
    std::size_t constexpr N = 200;
    std::size_t constexpr num_integrator_steps = 3;
    double constexpr time_step = 0.1;

    CasadiImplicitDae dae;
    CasadiImplicitDaeS dae_s;
    tmpc::StaticImplicitRungeKutta<double, 3, NX, NZ, NU> irk {tmpc::GaussRadauIIAMethod {3}};

    blaze::StaticVector<double, NX> const x0 {0.0, M_PI + 1., 0.0, 0.0};
    blaze::StaticVector<double, NU> const u {0.};
    blaze::StaticVector<double, NX> x = x0;
    blaze::StaticMatrix<double, NX + NZ, NX + NU> Sxu, Sf;

    std::cout << std::scientific << trans(x);
    for (std::size_t i = 0; i < N; ++i)
    {
        Sxu = 0.;
        blaze::diagonal(Sxu) = 1.;
        tmpc::integrate(irk, dae, dae_s, 0., time_step, num_integrator_steps, x, Sxu, u, x, Sf);
        std::cout << std::scientific << trans(x);
    }

    std::cout << "S_forw, sensitivities of simulation result wrt x,u:\n" << Sf;

    return 0;
}

