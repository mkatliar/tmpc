#include <tmpc/integrator/ImplicitRungeKutta.hpp>
#include <tmpc/integrator/StaticImplicitRungeKutta.hpp>
#include <tmpc/integrator/BackwardEulerMethod.hpp>
#include <tmpc/integrator/GaussLegendreMethod.hpp>

#include <tmpc/Testing.hpp>


namespace tmpc :: testing
{
	/// @brief Test integration of a simple linear ODE.
	class IrkSimpleLinearOdeTest
	:	public Test
	{
	protected:
		static size_t constexpr NX = 1;
		static size_t constexpr NZ = 0;
		static size_t constexpr NU = 1;
		using Real = double;

		template <typename Integrator>
		void testIntegrate_impl(Integrator const& integrator, double abs_tol, double rel_tol)
		{
			using VecX = blaze::StaticVector<Real, NX, blaze::columnVector>;
			using VecU = blaze::StaticVector<Real, NU, blaze::columnVector>;

			auto ode = [] (Real t, auto const& xdot, auto const& x, auto const& z,
				auto const& u, auto& f, auto& Jxdot, auto& Jx, auto& Jz)
			{
				f = x * u - xdot;
				Jxdot = {{-1.}};
				Jx = {{u[0]}};
			};

			Real const t0 = 0.11;
			VecX const x0 {3.};
			VecU const u {2.};
			Real const h = 0.025;
			VecX x1;
			integrator(ode, t0, h, x0, u, x1);

			TMPC_EXPECT_APPROX_EQ(x1, x0 * exp(u * h), abs_tol, rel_tol);
		}
	};


	TEST_F(IrkSimpleLinearOdeTest, testBackwardEulerDynamic)
	{
		testIntegrate_impl(ImplicitRungeKutta<Real> {BackwardEulerMethod {}, NX, NZ, NU}, 0., 0.002);
	}


	TEST_F(IrkSimpleLinearOdeTest, testGaussLegendre2Dynamic)
	{
		testIntegrate_impl(ImplicitRungeKutta<Real> {GaussLegendreMethod {2}, NX, NZ, NU}, 0., 1e-7);
	}


	TEST_F(IrkSimpleLinearOdeTest, testGaussLegendre3Dynamic)
	{
		testIntegrate_impl(ImplicitRungeKutta<Real> {GaussLegendreMethod {3}, NX, NZ, NU}, 0., 1e-14);
	}

	TEST_F(IrkSimpleLinearOdeTest, testBackwardEulerStatic)
	{
		testIntegrate_impl(StaticImplicitRungeKutta<Real, 1, NX, NZ, NU> {BackwardEulerMethod {}}, 0., 0.002);
	}


	TEST_F(IrkSimpleLinearOdeTest, testGaussLegendre2Static)
	{
		testIntegrate_impl(StaticImplicitRungeKutta<Real, 2, NX, NZ, NU> {GaussLegendreMethod {2}}, 0., 1e-7);
	}


	TEST_F(IrkSimpleLinearOdeTest, testGaussLegendre3Static)
	{
		testIntegrate_impl(StaticImplicitRungeKutta<Real, 3, NX, NZ, NU> {GaussLegendreMethod {3}}, 0., 1e-14);
	}
}