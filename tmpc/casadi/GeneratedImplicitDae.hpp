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

#include <blaze/Math.h>


namespace tmpc :: casadi
{
    /**
    * @brief Wraps a CasADi generated function for an implicit DAE
    * in a callable object that can be passed to tmpc integrators.
    *
    */
    class GeneratedImplicitDae
    {
    public:
        /**
        * @brief Constructor
        *
        * @param f_dae pointers to CasADi generated function. The CasADi function should have the following signature:
        * [t, xdot, x, z, u] -> [f, Jxdot, Jx, Jz]
        */
        explicit GeneratedImplicitDae(casadi_functions const * f_dae)
        :   casadiDae_ {f_dae}
        {
        }


        /**
         * @brief Calculate the DAE function and its sensitivities.
         *
         * @tparam Real real number type
         * @tparam VT0 vector type for the derivative of differential state
         * @tparam VT1 vector type for the differential state
         * @tparam VT2 vector type for the algebraic state
         * @tparam VT3 vector type for the control input
         * @tparam VT4 vector type for the right-hand side of the implicit DAE: 0 = f(t, xdot, x, z, u)
         * @tparam MT5 matrix type for the Jacobian of @a f w.r.t. @a xdot
         * @tparam SO5 storage order for @a MT5
         * @tparam MT6 matrix type for the Jacobian of @a f w.r.t. @a x
         * @tparam SO6 storage order for @a MT6
         * @tparam MT7 matrix type for the Jacobian of @a f w.r.t. @a z
         * @tparam SO7 storage order for @a MT7
         *
         * @param t time
         * @param xdot time-derivative of the differential state
         * @param x differential state
         * @param z algebraic state
         * @param u control input
         * @param f [out] value of the implicit ODE function
         * @param Jxdot [out] the Jacobian of @a f w.r.t. @a xdot
         * @param Jx [out] the Jacobian of @a f w.r.t. @a x
         * @param Jz [out] the Jacobian of @a f w.r.t. @a z
         */
        template <
            typename Real,
            typename VT0,
            typename VT1,
            typename VT2,
            typename VT3,
            typename VT4,
            typename MT5, bool SO5,
            typename MT6, bool SO6,
            typename MT7, bool SO7
        >
        void operator()(
            Real t,
            blaze::Vector<VT0, blaze::columnVector> const& xdot,
            blaze::Vector<VT1, blaze::columnVector> const& x,
            blaze::Vector<VT2, blaze::columnVector> const& z,
            blaze::Vector<VT3, blaze::columnVector> const& u,
            blaze::Vector<VT4, blaze::columnVector>& f,
            blaze::Matrix<MT5, SO5>& Jxdot,
            blaze::Matrix<MT6, SO6>& Jx,
            blaze::Matrix<MT7, SO7>& Jz
        ) const
        {
            casadiDae_(std::tie(t, xdot, x, z, u), std::tie(f, Jxdot, Jx, Jz));
        }

    private:
        tmpc::casadi::GeneratedFunction casadiDae_;
    };
}