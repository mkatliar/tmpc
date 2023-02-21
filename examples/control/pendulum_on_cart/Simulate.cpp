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

#include <pendulum_model.h>

#include <tmpc/casadi/GeneratedImplicitDae.hpp>
#include <tmpc/integrator/ImplicitRungeKutta.hpp>
#include <tmpc/integrator/GaussRadauIIAMethod.hpp>

#include <iostream>


int main(int, char **)
{
    std::size_t constexpr NX = 4, NU = 1;
    std::size_t constexpr N = 200;
    std::size_t constexpr num_integrator_steps = 3;
    double constexpr time_step = 0.1;

    tmpc::casadi::GeneratedImplicitDae dae {pendulum_ode_functions()};
    tmpc::ImplicitRungeKutta<double> irk {tmpc::GaussRadauIIAMethod {3}, NX, 0, NU};

    blaze::StaticVector<double, NX> const x0 {0.0, M_PI + 1., 0.0, 0.0};
    blaze::StaticVector<double, NU> const u {0.};
    blaze::StaticVector<double, NX> x = x0;

    std::cout << std::scientific << trans(x);
    for (std::size_t i = 0; i < N; ++i)
    {
        tmpc::integrate(irk, dae,	0., time_step, num_integrator_steps, x, u, x);
        std::cout << std::scientific << trans(x);
    }

    return 0;
}

