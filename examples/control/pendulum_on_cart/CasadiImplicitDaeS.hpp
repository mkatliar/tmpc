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

#include <tmpc/casadi/GeneratedFunction.hpp>


/**
 * @brief Implicit DAE system model with sensitivities
 * based on CasADi implementation.
 *
 */
class CasadiImplicitDaeS
{
public:
    CasadiImplicitDaeS();

    /**
    * @brief Calculate the DAE function and its sensitivities:
    *   dae_s(t, xdot, x, Sx, z, u, J);
    *
    * @tparam Real real number type
    * @tparam VT0 vector type for the derivative of differential state
    * @tparam VT1 vector type for the differential state
    * @tparam MT0 matrix type for input sensitivities
    * @tparam SO0 storage order for input sensitivities
    * @tparam VT2 vector type for the algebraic state
    * @tparam VT3 vector type for the control input
    * @tparam MT1 matrix type for the Jacobian
    * @tparam SO1 storage order for the Jacobian
    *
    * @param t time
    * @param xdot time-derivative of the differential state
    * @param x differential state
    * @param Sx input sensitivities dx/d(v,u), v \in R^NX
    * @param z algebraic state
    * @param u control input
    * @param Sf [out] sensitivities of the DAE function:
    *   Sf = (df/dx)*Sx + [0, df/du] = [df/dx*dx/dv, df/dx*dx/du + df/du]
    */
    template <
        typename Real,
        typename VT0,
        typename VT1,
        typename MT0, bool SO0,
        typename VT2,
        typename VT3,
        typename MT1, bool SO1
    >
    void operator()(
        Real t,
        blaze::Vector<VT0, blaze::columnVector> const& xdot,
        blaze::Vector<VT1, blaze::columnVector> const& x,
        blaze::Matrix<MT0, SO0> const& Sx,
        blaze::Vector<VT2, blaze::columnVector> const& z,
        blaze::Vector<VT3, blaze::columnVector> const& u,
        blaze::Matrix<MT1, SO1>& Sf
    ) const
    {
        casadiDae_(std::tie(t, xdot, x, Sx, z, u), std::tie(Sf));
    }

private:
    tmpc::casadi::GeneratedFunction casadiDae_;
};
