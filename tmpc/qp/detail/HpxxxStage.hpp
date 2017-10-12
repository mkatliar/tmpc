#pragma once

#include <tmpc/ocp/OcpSize.hpp>
#include <tmpc/ocp/OcpSolutionBase.hpp>
#include <tmpc/qp/OcpQpBase.hpp>

#include <tmpc/Matrix.hpp>
#include <tmpc/Math.hpp>

#include "UnpaddedMatrix.hpp"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <vector>
#include <array>


namespace tmpc :: detail
{
	///
	/// Common class for both HPMPC and HPIPM stage data.
	///
	template <typename Kernel_, StorageOrder SO>
	class HpxxxStage
	:	public OcpSolutionBase<HpxxxStage<Kernel_, SO>>
	,	public OcpQpBase<HpxxxStage<Kernel_, SO>>
	{
	public:
		using Kernel = Kernel_;
		using Real = typename Kernel::Real;

		HpxxxStage(OcpSize const& sz, size_t nx_next)
		:	size_(sz)
		,	hidxb_(sz.nu() + sz.nx())
		,	Q_ {sz.nx(), sz.nx()}
		,	R_ {sz.nu(), sz.nu()}
		,	S_ {sz.nu(), sz.nx()}	// <-- HPMPC convention for S is [nu, nx] (the corresponding cost term is u' * S_{hpmpc} * x)
		,	q_(sz.nx())
		,	r_(sz.nu())
		,	A_ {nx_next, sz.nx()}
		,	B_ {nx_next, sz.nu()}
		,	b_(nx_next)
		,	C_ {sz.nc(), sz.nx()}
		,	D_ {sz.nc(), sz.nu()}
		,	lb_(sz.nu() + sz.nx())
		,	ub_(sz.nu() + sz.nx())
		,	lb_internal_(sz.nu() + sz.nx())
		,	ub_internal_(sz.nu() + sz.nx())
		,	lbd_(sz.nc())
		,	ubd_(sz.nc())
		,	x_(sz.nx())
		,	u_(sz.nu())
		,	pi_(nx_next)
		,	lam_(2 * sz.nc() + 2 * (sz.nx() + sz.nu()))
		{
			// Initialize all numeric data to NaN so that if an uninitialized object
			// by mistake used in calculations is easier to detect.
			this->setNaN();

			// hidxb is initialized to its maximum size, s.t. nb == nx + nu.
			// This is necessary so that the solver workspace memory is calculated as its maximum when allocated.
			int n = 0;
			std::generate(hidxb_.begin(), hidxb_.end(), [&n] { return n++; });
		}

		HpxxxStage(HpxxxStage const&) = default;
		HpxxxStage(HpxxxStage &&) = default;

		auto const& A() const { return A_; }
		template <typename T> void A(const T& a) { noresize(A_) = a; }

		auto const& B() const { return B_; }
		template <typename T> void B(const T& b) { noresize(B_) = b; }

		auto const& b() const { return b_; }
		template <typename T> void b(const T& b) { noresize(b_) = b; }

		auto const& C() const { return C_; }
		template <typename T> void C(const T& c) { noresize(C_) = c; }

		auto const& D() const { return D_; }
		template <typename T> void D(const T& d) { noresize(D_) = d; }

		auto const& lbd() const {	return lbd_; }
		template <typename T> void lbd(const T& lbd) { noresize(lbd_) = lbd; }

		auto lbu() const { return subvector(lb_, 0, size_.nu());	}		
		
		// TODO: consider setting both upper and lower bounds at the same time.
		// Maybe create a Bounds class?
		template <typename T> void lbu(const T& lbu) 
		{ 
			subvector(lb_, 0, size_.nu()) = lbu;
		}

		auto lbx() const { return subvector(lb_, size_.nu(), size_.nx()); }
		template <typename T> void lbx(const T& lbx) 
		{ 
			subvector(lb_, size_.nu(), size_.nx()) = lbx; 
		}

		auto const& Q() const { return Q_; }
		template <typename T> void Q(const T& q) { noresize(Q_) = q; }

		auto const& R() const { return R_; }
		template <typename T> void R(const T& r) { noresize(R_) = r; }

		// HPMPC convention for S is [nu, nx], therefore the trans().
		auto S() const { return trans(S_); }
		void S(Real v) { S_ = v; }
		template <typename T> void S(const T& s) { noresize(S_) = trans(s); }

		auto const& q() const { return q_; }
		template <typename T> void q(const T& q) { noresize(q_) = q; }

		auto const& r() const { return r_; }
		template <typename T> void r(const T& r) { noresize(r_) = r; }

		auto const& ubd() const { return ubd_; }
		template <typename T> void ubd(const T& ubd) { noresize(ubd_) = ubd; }

		auto ubu() const { return subvector(ub_, 0, size_.nu()); }
		template <typename T> void ubu(const T& ubu) 
		{ 
			subvector(ub_, 0, size_.nu()) = ubu;
		}

		auto ubx() const { return subvector(ub_, size_.nu(), size_.nx()); }
		template <typename T> void ubx(const T& ubx) 
		{ 
			subvector(ub_, size_.nu(), size_.nx()) = ubx; 
		}

		auto const& x() const { return x_; }
		auto const& u() const { return u_;	}
		auto const& pi() const	{ return pi_; }
		
		auto lam_lbu() const 
		{ 
			return subvector(lam_, 2 * size_.nc(), size_.nu()); 
		}

		auto lam_ubu() const 
		{ 
			return subvector(lam_, 2 * size_.nc() + size_.nu(), size_.nu()); 
		}

		auto lam_lbx() const 
		{ 
			return subvector(lam_, 2 * size_.nc() + 2 * size_.nu(), size_.nx()); 
		}

		auto lam_ubx() const 
		{ 
			return subvector(lam_, 2 * size_.nc() + 2 * size_.nu() + size_.nx(), size_.nx()); 
		}

		auto lam_lbd() const 
		{ 
			return subvector(lam_, 0, size_.nc()); 
		}

		auto lam_ubd() const 
		{ 
			return subvector(lam_, size_.nc(), size_.nc()); 
		}

		OcpSize const& size() const { return size_; }

		// Adjust hidxb so to account for infs in state and input bounds.
		void adjustBoundsIndex()
		{
			// this will not change the capacity and the data() pointers should stay the same.
			hidxb_.clear();
			lb_internal_.clear();
			ub_internal_.clear();

			// Cycle through the bounds and check for infinities
			for (size_t i = 0; i < size_.nu() + size_.nx(); ++i)
			{
				if (std::isfinite(lb_[i]) && std::isfinite(ub_[i]))
				{
					// If both bounds are finite, add i to the bounds index,
					// and copy values to the lb_internal_ and ub_internal_.
					hidxb_.push_back(i);
					lb_internal_.push_back(lb_[i]);
					ub_internal_.push_back(ub_[i]);
				}
				else 
				{
					// Otherwise, check that the values are [-inf, inf]
					if (!(lb_[i] == -inf<Real>() && ub_[i] == inf<Real>()))
						throw std::invalid_argument("And invalid QP bound is found. For HPMPC, "
							"the bounds should be either both finite or [-inf, inf]");
				}
			}
		}

		// ******************************************************
		//                HPMPC raw data interface.
		//
		// The prefixes before _data() correspond to the names of
		// the argument to c_order_d_ip_ocp_hard_tv().
		// ******************************************************
		Real const * A_data () const { return A_.data(); }
		Real const * B_data () const { return B_.data(); }
		Real const * b_data () const { return b_.data();	}
		Real const * Q_data () const { return Q_.data(); }
		Real const * S_data () const { return S_.data(); }
		Real const * R_data () const { return R_.data(); }
		Real const * q_data () const { return q_.data(); }
		Real const * r_data () const { return r_.data();	}
		Real const * lb_data() const { return lb_internal_.data(); }
		Real const * ub_data() const { return ub_internal_.data(); }
		Real const * C_data () const { return C_.data(); }
		Real const * D_data () const { return D_.data(); }
		Real const * lg_data() const { return lbd_.data(); }
		Real const * ug_data() const { return ubd_.data(); }
		int const * hidxb_data() const { return hidxb_.data(); }
		int nb() const { return hidxb_.size(); }

		Real * x_data() { return x_.data(); }
		Real * u_data() { return u_.data(); }
		Real * pi_data() { return pi_.data(); }
		Real * lam_data() { return lam_.data(); }
		Real * lam_lb_data() { return lam_.data(); }
		Real * lam_ub_data() { return lam_lb_data() + size_.nx() + size_.nu(); }
		Real * lam_lg_data() { return lam_ub_data() + size_.nx() + size_.nu(); }
		Real * lam_ug_data() { return lam_lg_data() + size_.nc(); }

	private:
		OcpSize size_;

		// Some magic data for HPMPC
		std::vector<int> hidxb_;

		// Hessian = [R, S; S', Q]
		UnpaddedMatrix<Kernel, SO> Q_;
		UnpaddedMatrix<Kernel, SO> R_;
		UnpaddedMatrix<Kernel, SO> S_;			

		// Gradient = [r; q]
		DynamicVector<Kernel> r_;
		DynamicVector<Kernel> q_;

		// Inter-stage equalities x_{k+1} = A x_k + B u_k + c_k
		UnpaddedMatrix<Kernel, SO> A_;
		UnpaddedMatrix<Kernel, SO> B_;
		DynamicVector<Kernel> b_;

		// Inequality constraints d_{min} <= C x_k + D u_k <= d_{max}
		UnpaddedMatrix<Kernel, SO> C_;
		UnpaddedMatrix<Kernel, SO> D_;
		DynamicVector<Kernel> lbd_;
		DynamicVector<Kernel> ubd_;

		// Bound constraints:
		// lb <= [u; x] <= ub
		DynamicVector<Kernel> lb_;
		DynamicVector<Kernel> ub_;

		// Lower and upper bound arrays for HPMPC,
		// containing finite values only.
		std::vector<Real> lb_internal_;
		std::vector<Real> ub_internal_;

		DynamicVector<Kernel> x_;
		DynamicVector<Kernel> u_;
		DynamicVector<Kernel> pi_;
		DynamicVector<Kernel> lam_;
	};
}
