#pragma once

#include <tmpc/Matrix.hpp>

namespace tmpc
{
	template <typename Real_>
	class MpcTrajectoryPoint
	{
	public:
		using Real = Real_;

		MpcTrajectoryPoint(size_t NX, size_t NU)
		:	x_(NX)
		,	u_(NU)
		{
		}

		template <typename VectorX, typename VectorU>
		MpcTrajectoryPoint(VectorX const& x, VectorU const& u)
		:	x_(x)
		,	u_(u)
		{
		}

		template <typename T>
		void x(T const& val)
		{
			x_ = val;
		}

		DynamicVector<Real> const& x() const
		{
			return x_;
		}

		template <typename T>
		void u(T const& val)
		{
			u_ = val;
		}

		DynamicVector<Real> const& u() const
		{
			return u_;
		}

	private:
		DynamicVector<Real> x_;
		DynamicVector<Real> u_;
	};

	template <typename Real>
	inline std::ostream& operator<<(std::ostream& os, MpcTrajectoryPoint<Real> const& p)
	{
		os << "x=" << trans(p.x()) << "\tu=" << trans(p.u());
	}
}
