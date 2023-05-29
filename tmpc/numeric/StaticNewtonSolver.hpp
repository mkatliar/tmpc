#pragma once

#include <tmpc/Math.hpp>
#include <tmpc/SizeT.hpp>
#include <tmpc/Exception.hpp>

#include <array>


namespace tmpc
{
    /// @brief Solves statically sized systems of non-linear equations with Newton method.
    ///
    /// @tparam Real real number type
    /// @tparam NX number of variables
    ///
    template <typename Real, size_t NX>
    class StaticNewtonSolver
    {
    public:
        explicit StaticNewtonSolver()
        :   J_(Real {})
        {
        }


        template <typename F, typename VT1, typename VT2>
        void operator()(F const& fun,
            blaze::Vector<VT1, blaze::columnVector> const& x0,
            blaze::Vector<VT2, blaze::columnVector>& xf)
        {
            (*this)(fun, x0, xf, emptyMonitor());
        }


        template <typename F, typename VT1, typename VT2, typename Monitor>
        void operator()(F const& fun,
            blaze::Vector<VT1, blaze::columnVector> const& x0,
            blaze::Vector<VT2, blaze::columnVector>& xf,
            Monitor&& monitor)
        {
            if (size(x0) != NX)
                TMPC_THROW_EXCEPTION(std::invalid_argument("Invalid vector dimension"));

            functionEvaluations_ = 0;

            *xf = *x0;
            evaluate(fun, *xf, r_, J_, monitor);
            factorizeJacobian();

            for (iterations_ = 0; (residualMaxNorm_ = maxNorm(r_)) >= residualTolerance_ && iterations_ < maxIterations_; ++iterations_)
			{
                // Calculate search direction d(n)=-inv(J(n))*r(n)
                jacobianSolve(r_, d_);

                // Step size
                Real t = 1.;

                // Do backtracking search.
                // alpha == 1. disables backtracking.
                while (evaluate(fun, x1_ = *xf + t * d_, r1_, J_, monitor), alpha_ < 1. && !residualDecreased(r1_, r_))
                    t *= alpha_;
                factorizeJacobian();

                // Netwon method update: x(n+1) = x(n) + t*d
                *xf = x1_;
                r_ = r1_;
            }

            if (residualMaxNorm_ >= residualTolerance_)
                TMPC_THROW_EXCEPTION(std::runtime_error {"Newton residual beyond tolerance after max number of iterations"});
        }


        /// @brief Solve the equation using Newton method and calculate
        /// solution sensitivities w.r.t. parameters.
        ///
        /// @param dxf_dp a NX-by-NP matrix of solution sensitivities
        ///
        template <
            typename F,
            typename DFDP,
            typename VT1,
            typename VT2,
            typename MT
        >
        void operator()(F const& fun,
            DFDP const& dfdp,
            blaze::Vector<VT1, blaze::columnVector> const& x0,
            blaze::DenseVector<VT2, blaze::columnVector>& xf,
            blaze::DenseMatrix<MT, blaze::columnMajor>& dxf_dp)
        {
            if (size(x0) != NX)
                TMPC_THROW_EXCEPTION(std::invalid_argument {"Invalid vector dimension"});

            (*this)(fun, *x0, *xf);

            // Calculate df(x,p)/dp at the solution
            dfdp(*xf, *dxf_dp);

            // From 0 = df(x,p)/dx * dx^*/dp + df(x,p)/dp
            // calculate dx^*/dp = -inv(df(x,p)/dx) * df(x,p)/dp
            jacobianSolve(*dxf_dp, *dxf_dp);
        }


        size_t maxIterations() const noexcept
        {
            return maxIterations_;
        }


        void maxIterations(size_t val)
        {
            maxIterations_ = val;
        }


        Real residualTolerance() const noexcept
        {
            return residualTolerance_;
        }


        void residualTolerance(Real val)
        {
            if (val < 0)
                TMPC_THROW_EXCEPTION(std::invalid_argument {"Residual tolerance must be non-negative"});

            residualTolerance_ = val;
        }


        Real residualMaxNorm() const noexcept
        {
            return residualMaxNorm_;
        }


        /// @brief Total number of Newton iterations during last solve.
        size_t iterations() const noexcept
        {
            return iterations_;
        }


        /// @brief Total number of function evaluations during last solve.
        size_t functionEvaluations() const noexcept
        {
            return functionEvaluations_;
        }


        Real backtrackingAlpha() const noexcept
        {
            return alpha_;
        }


        /// @brief Set backtracking alpha parameter value.
        ///
        /// alpha must be withing the range (0., 1.].
        /// Setting alpha = 1. disables backtracking.
        void backtrackingAlpha(Real val)
        {
            if (!(0. < val && val <= 1.))
                TMPC_THROW_EXCEPTION(std::invalid_argument {"Backtracking alpha must be within the (0, 1] range"});

            alpha_ = val;
        }


        /// @brief Solve J*x+b=0 w.r.t. x where x, b are vectors and J is the current Jacobian.
        template <typename VT1, typename VT2>
        void jacobianSolve(
            blaze::DenseVector<VT1, blaze::columnVector> const& b,
            blaze::DenseVector<VT2, blaze::columnVector>& x) const
        {
            *x = -*b;
            getrs(*J_, *x, 'N', ipiv_.data());
        }


        /// @brief Solve J*X+B=0 w.r.t. x where X, B are matrices and J is the current Jacobian.
        template <typename MT1, typename MT2>
        void jacobianSolve(
            blaze::DenseMatrix<MT1, blaze::columnMajor> const& B,
            blaze::DenseMatrix<MT2, blaze::columnMajor>& X) const
        {
            *X = -*B;
            getrs(*J_, *X, 'N', ipiv_.data());
        }


    private:
        blaze::StaticVector<Real, NX, blaze::columnVector> x1_;
        blaze::StaticVector<Real, NX, blaze::columnVector> r_;
        blaze::StaticVector<Real, NX, blaze::columnVector> r1_;
        blaze::StaticVector<Real, NX, blaze::columnVector> d_;

        // Factorized Jacobian.
        // Must be column-major in order blaze::getrf() and blaze::getrs() to work as expected.
        blaze::StaticMatrix<Real, NX, NX, blaze::columnMajor> J_;

        size_t iterations_ = 0;
		size_t maxIterations_ = 10;
        size_t functionEvaluations_ = 0;
		Real residualMaxNorm_ = inf<Real>();
        Real residualTolerance_ = 1e-10;

        // Backtracking alpha
        Real alpha_ = 1.;

        mutable std::array<int, NX> ipiv_;


        template <typename F, typename VT0, typename VT1, typename MT, typename Monitor>
        requires blaze::IsDenseVector_v<VT0> && blaze::IsDenseVector_v<VT1> && blaze::IsDenseMatrix_v<MT>
        void evaluate(F&& fun, VT0 const& x, VT1& r, MT& J, Monitor&& monitor)
        {
            fun(x, r, J);
            ++functionEvaluations_;
            monitor(iterations_, std::as_const(x), std::as_const(r), std::as_const(J));
        }


        template <typename VT1, typename VT2, bool TF>
        bool residualDecreased(blaze::Vector<VT1, TF> const& r1, blaze::Vector<VT2, TF> const& r0) const
        {
            size_t const n = size(r1);

            if (n != size(r0))
                TMPC_THROW_EXCEPTION(std::invalid_argument("Vector sizes don't match"));

            size_t i = 0;
            while (i < n && abs((*r1)[i]) < residualTolerance_ || abs((*r1)[i]) < abs((*r0)[i]))
                ++i;

            return i >= n;
        }


        void factorizeJacobian()
        {
            // Factorize the Jacobian
            getrf(J_, ipiv_.data());
        }


        static auto emptyMonitor()
        {
            return [] (size_t, auto const&, auto const&, auto const&) {};
        }
    };
}