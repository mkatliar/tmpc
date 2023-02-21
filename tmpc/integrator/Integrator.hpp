#pragma once

#include <tmpc/SizeT.hpp>

#include <blaze/Math.h>

#include <concepts>


namespace tmpc
{
    /// @brief CRTP base class for integrators.
    ///
    /// TODO: deprecate this class and use concepts for integrator objects.
    ///
    template <typename Derived>
    class Integrator
    {
    public:
        Derived const& operator~() const noexcept
        {
            return static_cast<Derived const&>(*this);
        }


        Derived& operator~() noexcept
        {
            return static_cast<Derived&>(*this);
        }


    protected:
        Integrator() = default;
        Integrator(Integrator const&) = default;
        Integrator(Integrator&&) = default;
        Integrator& operator=(Integrator const&) = default;
        Integrator& operator=(Integrator&&) = default;
        ~Integrator() = default;
    };


    /**
     * @brief Perform a required number of integration steps with a given integrator.
     *
     * @tparam I integrator type
     * @tparam DE type for the differential equation callable object
     * @tparam Real real number type
     * @tparam VT1 vector type for initial state
     * @tparam VT2 vector type for control input
     * @tparam VT3 vector type for final state
     *
     * @param integrator integrator object
     * @param de a callable that represents the differential equation. Can be ODE or DAE, depending on the integrator type.
     * @param t0 start time
     * @param h time step. The final state @a xf is calculated for @a t0 + @a h.
     * @param num_integrator_steps Number of integration steps to perform. The interval of length @h is divided into
     *  @a num_integrator_steps equal subintervals, and the integrator is called on each of them in sequence.
     * @param x0 initial state
     * @param u control input
     * @param xf final state (result)
     */
    template <typename I, typename DE, typename Real, typename VT1, typename VT2, typename VT3>
        requires(std::invocable<I, DE const&, Real, Real, VT3 const&, VT2 const&, VT3&>)
    inline void integrate(
        I const& integrator,
        DE const& de,
        Real t0, Real h, size_t num_integrator_steps,
        blaze::Vector<VT1, blaze::columnVector> const& x0,
        blaze::Vector<VT2, blaze::columnVector> const& u,
        blaze::Vector<VT3, blaze::columnVector>& xf)
    {
        // Actual integrator step
        Real const integrator_step = h / num_integrator_steps;

        *xf = *x0;
        for (size_t i = 0; i < num_integrator_steps; ++i)
            integrator(de, t0 + integrator_step * i, integrator_step, *xf, *u, *xf);
    }
}