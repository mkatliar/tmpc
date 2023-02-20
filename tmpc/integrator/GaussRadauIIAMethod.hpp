#pragma once

#include <tmpc/Exception.hpp>

#include <blaze/Math.h>

#include <cmath>


namespace tmpc
{
    /// @brief Gauss-Radau IIA method with a specified number of stages.
    struct GaussRadauIIAMethod
    {
        explicit GaussRadauIIAMethod(size_t n_stages)
        :   stages_(n_stages)
        {
        }


        template <typename MT, bool SO, typename VT1, typename VT2>
        void butcherTableau(
            blaze::Matrix<MT, SO>& A,
            blaze::Vector<VT1, blaze::rowVector>& b,
            blaze::Vector<VT2, blaze::columnVector>& c) const
        {
            switch (stages_)
            {
                case 2:
                    // The tables for the 2- and 3-point Gauss-Radau method are taken from
                    // https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Radau_IIA_methods
                    *A = {
                        {5. / 12., -1. / 12.},
                        {3. / 4., 1. / 4.}
                    };
                    *b = {3. / 4., 1. / 4.};
                    *c = {1. / 3., 1.};

                    break;

                case 3:
                    *A = {
                        {11. / 45. - 7. * sqrt(6.) / 360., 37. / 225. - 169. * sqrt(6.) / 1800., -2. / 225. + sqrt(6.) / 75.},
                        {37. / 225. + 169. * sqrt(6.) / 1800., 11. / 45. + 7. * sqrt(6.) / 360., -2. / 225. - sqrt(6.) / 75.},
                        {4. / 9. - sqrt(6.) / 36., 4. / 9. + sqrt(6.) / 36., 1. / 9.},
                    };
                    *b = {4. / 9. - sqrt(6.) / 36., 4. / 9. + sqrt(6.) / 36., 1. / 9.};
                    *c = {2. / 5. - sqrt(6.) / 10., 2. / 5. + sqrt(6.) / 10., 1.};

                    break;

                default:
                    TMPC_THROW_EXCEPTION(std::invalid_argument("Unsupported number of Gauss-Radau IIA method stages"));
            }
        }


        size_t stages() const
        {
            return stages_;
        }


    private:
        size_t const stages_;
    };
}