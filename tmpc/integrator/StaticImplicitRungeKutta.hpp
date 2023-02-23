#pragma once

#include <cstddef>
#include <tmpc/integrator/ImplicitIntegrator.hpp>
#include <tmpc/numeric/StaticNewtonSolver.hpp>

#include <tmpc/Exception.hpp>

#include <stdexcept>


namespace tmpc
{
	/**
	 * \defgroup integrators Integrators
	 */

	/**
	 * \brief Implicit Runge-Kutta integrator for statically-sized problem.
	 * \ingroup integrators
	 *
	 * @tparam Real real number type
	 * @tparam M number of integration method stages
	 * @tparam NX number of differential states
	 * @tparam NZ number of algebraic states
	 * @tparam NY number of outputs
	 *
	 */
	template <
		typename Real,
		std::size_t M,
		std::size_t NX,
		std::size_t NZ,
		std::size_t NU,
		std::size_t NY = 0
	>
	class StaticImplicitRungeKutta
	:	public ImplicitIntegrator<StaticImplicitRungeKutta<Real, M, NX, NZ, NU, NY>>
	{
		static std::size_t constexpr NW = NX + NZ;

	public:
		/**
		 * @brief Constructor
		 *
		 * @tparam Method type for integration method
		 * @param method defines Butcher table for a specific integration method
		 */
		template <typename Method>
		StaticImplicitRungeKutta(Method const& method)
		:	kz_(Real {})
		{
			method.butcherTableau(A_, b_, c_);
		}


		/**
		 * @brief Integrate a DAE over a time interval.
		 *
		 * @tparam DAE DAE type
		 * @tparam VT1 initial state vector type
		 * @tparam VT2 control input vector type
		 * @tparam VT3 final state vector type
		 *
		 * @param dae DAE functor. It has the following signature:
		 * 		dae(t, xdot, x, z, u, f, Jxdot, Jx, Jz);
		 *		where t, xdot, x, z, u are input arguments,
		 *		f, Jxdot, Jx, Jz are output arguments.
		 * @param t0 time at which integration starts
		 * @param h time interval for integration
		 * @param x0 initial state
		 * @param u control input
		 * @param xf final state
		 */
		template <typename DAE, typename VT1, typename VT2, typename VT3>
		void operator()(DAE const& dae, Real t0, Real h,
			blaze::Vector<VT1, blaze::columnVector> const& x0,
			blaze::Vector<VT2, blaze::columnVector> const& u,
			blaze::Vector<VT3, blaze::columnVector>& xf) const
		{
			(*this)(dae, t0, h, *x0, *u, *xf, [] (size_t, auto const&, auto const&, auto const&) {});
		}


		/**
		 * @brief Integrate a DAE over a time interval.
		 *
		 * @tparam DAE DAE type
		 * @tparam VT1 initial state vector type
		 * @tparam VT2 control input vector type
		 * @tparam VT3 final state vector type
		 * @tparam Newton solver monitor object type
		 *
		 * @param dae DAE functor. It has the following signature:
		 * 		dae(t, xdot, x, z, u, f, Jxdot, Jx, Jz);
		 *		where t, xdot, x, z, u are input arguments,
		 *		f, Jxdot, Jx, Jz are output arguments.
		 * @param t0 time at which integration starts
		 * @param h time interval for integration
		 * @param x0 initial state
		 * @param u control input
		 * @param xf final state
		 * @param monitor Newton solver monitor object
		 */
		template <typename DAE, typename VT1, typename VT2, typename VT3, typename Monitor>
		void operator()(DAE const& dae, Real t0, Real h,
			blaze::Vector<VT1, blaze::columnVector> const& x0,
			blaze::Vector<VT2, blaze::columnVector> const& u,
			blaze::Vector<VT3, blaze::columnVector>& xf,
			Monitor monitor) const
		{
			if (size(x0) != NX)
				TMPC_THROW_EXCEPTION(std::invalid_argument("Invalid size of x0"));

			if (size(u) != NU)
				TMPC_THROW_EXCEPTION(std::invalid_argument("Invalid size of u"));

			// Finding the root of the following equation using Newton method:
			// 0 = f(t_0 + c_i h, x_0 + \sum_{j=1}^s a_{i,j} k_j) - k_i

			// If warm-starting, reuse the previous value of k as the starting point,
			// otherwise reset it to 0.
			if (!warmStart_)
				reset(kz_);

			newtonSolver_(newtonResidual(dae, t0, h, *x0, *u), kz_, kz_, monitor);

			// Calculating the value of the integral
			*xf = x0;
			for (size_t i = 0; i < M; ++i)
				*xf += h * b_[i] * subvector(kz_, i * NW, NX, blaze::unchecked);
		}


		template <
			typename DAE,
			typename DAE_S,
			typename VT1,
			typename MT1, bool SO1,
			typename VT2,
			typename VT3,
			typename MT2, bool SO2>
		void operator()(
			DAE const& dae,
			DAE_S const& dae_s,
			Real t0, Real h,
			blaze::Vector<VT1, blaze::columnVector> const& x0,
			blaze::Matrix<MT1, SO1> const& Sx,
			blaze::Vector<VT2, blaze::columnVector> const& u,
			blaze::Vector<VT3, blaze::columnVector>& xf,
			blaze::Matrix<MT2, SO2>& Sf) const
		{
			if (size(x0) != NX)
				TMPC_THROW_EXCEPTION(std::invalid_argument("Invalid size of x0"));

			if (size(u) != NU)
				TMPC_THROW_EXCEPTION(std::invalid_argument("Invalid size of u"));

			// Finding the root of the following equation using Newton method:
			// 0 = f(t_0 + c_i h, x_0 + \sum_{j=1}^s a_{i,j} k_j) - k_i

			// If warm-starting, reuse the previous value of k as the starting point,
			// otherwise reset it to 0.
			if (!warmStart_)
				reset(kz_);

			// Calculate implicit DAE/DAE solution k and its sensitivities
			newtonSolver_(
				newtonResidual(dae, t0, h, *x0, *u),
				newtonParamSensitivity(dae_s, t0, h, *x0, *Sx, *u),
				kz_, kz_, K_);

			// Calculating sensitivities of intermediate state variables.
			for (size_t i = 0; i < M; ++i)
			{
				S(i) = *Sx;

				for (size_t j = 0; j < M; ++j)
					S(i) += h * A_(i, j) * submatrix(K_, j * NW, 0, NX, NX + NU, blaze::unchecked);
			}

			// Calculating the value of the integral and final state sensitivities
			*xf = x0;
			*Sf = *Sx;

			for (size_t i = 0; i < M; ++i)
			{
				*xf += h * b_[i] * subvector(kz_, i * NW, NX, blaze::unchecked);
				*Sf += h * b_[i] * submatrix(K_, i * NW, 0, NX, NX + NU, blaze::unchecked);
			}
		}


		template <
			typename DAE,
			typename DAE_S,
			typename Residual,
			typename VT1,
			typename MT1, bool SO1,
			typename VT2,
			typename VT3,
			typename MT2, bool SO2,
			typename VT4,
			typename MT3, bool SO3>
		void operator()(
			DAE const& dae,
			DAE_S const& dae_s,
			Residual const& res,
			Real t0, Real h,
			blaze::Vector<VT1, blaze::columnVector> const& x0,
			blaze::Matrix<MT1, SO1> const& Sx,
			blaze::Vector<VT2, blaze::columnVector> const& u,
			blaze::Vector<VT3, blaze::columnVector>& xf,
			blaze::Matrix<MT2, SO2>& Sf,
			Real& l,
			blaze::Vector<VT4, blaze::columnVector>& g,
			blaze::Matrix<MT3, SO3>& H) const
		{
			if (size(x0) != NX)
				TMPC_THROW_EXCEPTION(std::invalid_argument("Invalid size of x0"));

			if (size(u) != NU)
				TMPC_THROW_EXCEPTION(std::invalid_argument("Invalid size of u"));

			// Calculate the final state value and its sensitivities
			(*this)(dae, dae_s, t0, h, *x0, *Sx, *u, *xf, *Sf);

			// Calculate the integral of the Lagrange term, its gradient, and its Gauss-Newton Hessian
			for (size_t i = 0; i < M; ++i)
			{
				// Calculate the residual and its sensitivities at i-th intermediate point
				res(t0 + h * c_[i], s(i), S(i), z(i), Z(i), *u, r_, Jr_);

				// Update the cost, its gradient, and its Gauss-Newton Hessian
				l += h * b_[i] * sqrNorm(r_) / 2.;
				*g += h * b_[i] * trans(Jr_) * r_;
				*H += h * b_[i] * trans(Jr_) * Jr_;
			}
		}


		/// @brief Get Newton method residual tolerance
		Real newtonResidualTolerance() const
		{
			return newtonSolver_.residualTolerance();
		}


		/// @brief Set Newton method residual tolerance
		void newtonResidualTolerance(Real val)
		{
			newtonSolver_.residualTolerance(val);
		}


		/// @brief Get max number of Newton iterations
		size_t newtonMaxIterations() const
		{
			return newtonSolver_.maxIterations();
		}


		/// @brief Set max number of Newton iterations
		void newtonMaxIterations(size_t val)
		{
			newtonSolver_.maxIterations(val);
		}


		/// @brief Set Newton method backtracking parameter
		void newtonBacktrackingAlpha(Real val)
		{
			newtonSolver_.backtrackingAlpha(val);
		}


		/// @brief Get number of Newton iterations made on the last operator() call.
		size_t newtonIterations() const
		{
			return newtonSolver_.iterations();
		}


		/// @brief Get warm start value.
		bool warmStart() const
		{
			return warmStart_;
		}


		/// @brief Switch warm start on and off.
		void warmStart(bool val)
		{
			warmStart_ = val;
		}


	private:
		// Butcher Tableau
		blaze::StaticMatrix<Real, M, M> A_;
		blaze::StaticVector<Real, M, blaze::rowVector> b_;
		blaze::StaticVector<Real, M, blaze::columnVector> c_;

		// Intermediate state variables
		mutable blaze::StaticVector<Real, M * NX, blaze::columnVector> s_;

		// Sensitivity of s w.r.t. (x,u)
		mutable blaze::StaticMatrix<Real, M * NX, NX + NU> S_;

		mutable blaze::StaticMatrix<Real, NW, NX, blaze::columnMajor> df_dx_;
		mutable blaze::StaticVector<Real, M * NW, blaze::columnVector> kz_;

		// Sensitivity of K w.r.t. (x,u)
		mutable blaze::StaticMatrix<Real, M * NW, NX + NU, blaze::columnMajor> K_;

		// Holds the residual
		mutable blaze::StaticVector<Real, NY> r_;

		// Holds the Jacobian of the residual
		mutable blaze::StaticMatrix<Real, NY, NX + NU> Jr_;

		mutable StaticNewtonSolver<Real, M * NW> newtonSolver_;

		// Use previous solution as the initial point in the Newton method
		bool warmStart_ = false;


		template <typename DAE, typename VT1, typename VT2>
		auto newtonResidual(DAE const& dae, Real t0, Real h,
			blaze::Vector<VT1, blaze::columnVector> const& x0,
			blaze::Vector<VT2, blaze::columnVector> const& u) const
		{
			return [this, &dae, t0, h, &x0, &u] (auto const& kz, auto& r, auto& J)
			{
				for (size_t i = 0; i < M; ++i)
				{
					s(i) = *x0;
					for (size_t j = 0; j < M; ++j)
						s(i) += h * A_(i, j) * subvector(kz, j * NW, NX, blaze::unchecked);

					auto const k_i = blaze::subvector(kz, i * NW, NX, blaze::unchecked);
					auto const z_i = subvector(kz, i * NW + NX, NZ, blaze::unchecked);
					auto f = subvector(r, i * NW, NW, blaze::unchecked);
					auto Jxdot = submatrix(J, i * NW, i * NW, NW, NX, blaze::unchecked);
					auto Jz = submatrix(J, i * NW, i * NW + NX, NW, NZ, blaze::unchecked);
					dae(t0 + c_[i] * h, k_i, s(i), z_i, *u, f, Jxdot, df_dx_, Jz);

					for (size_t j = 0; j < M; ++j)
					{
						auto Jx_ij = submatrix(J, i * NW, j * NW, NW, NX, blaze::unchecked);
						auto Jz_ij = submatrix(J, i * NW, j * NW + NX, NW, NZ, blaze::unchecked);

						if (j != i)
						{
							Jx_ij = h * A_(i, j) * df_dx_;
							reset(Jz_ij);
						}
						else
						{
							Jx_ij += h * A_(i, j) * df_dx_;
						}
					}
				}
			};
		}


		template <
			typename DAE_S,
			typename VT1,
			typename MT, bool SO,
			typename VT2
		>
		auto newtonParamSensitivity(
			DAE_S const& dae_s,
			Real t0, Real h,
			blaze::Vector<VT1, blaze::columnVector> const& x0,
			blaze::Matrix<MT, SO> const& Sx,
			blaze::Vector<VT2, blaze::columnVector> const& u) const
		{
			return [this, &dae_s, t0, h, &Sx, &u] (auto const& kz, auto& df_dp)
			{
				for (size_t i = 0; i < M; ++i)
				{
					// NOTE: s(i) have valid values here,
					// which have already been calculated while calculating the Newton residual.
					auto const k_i = subvector(kz, i * NW, NX, blaze::unchecked);
					auto const z_i = subvector(kz, i * NW + NX, NZ, blaze::unchecked);
					auto dfi_dxu = submatrix(df_dp, i * NW, 0, NW, NX + NU, blaze::unchecked);
					dae_s(t0 + c_[i] * h, k_i, s(i), *Sx, z_i, *u, dfi_dxu);
				}
			};
		}


		decltype(auto) s(size_t i) const
		{
			return subvector(s_, NX * i, NX, blaze::unchecked);
		}


		decltype(auto) S(size_t i) const
		{
			return submatrix(S_, NX * i, 0, NX, NX + NU, blaze::unchecked);
		}


		decltype(auto) z(size_t i) const
		{
			return subvector(kz_, NW * i + NX, NZ, blaze::unchecked);
		}


		decltype(auto) Z(size_t i) const
		{
			return submatrix(K_, NW * i + NX, 0, NZ, NX + NU, blaze::unchecked);
		}
	};
}
