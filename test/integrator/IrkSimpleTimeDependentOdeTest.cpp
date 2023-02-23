#include <tmpc/integrator/DynamicImplicitRungeKutta.hpp>
#include <tmpc/integrator/StaticImplicitRungeKutta.hpp>
#include <tmpc/integrator/BackwardEulerMethod.hpp>
#include <tmpc/integrator/GaussLegendreMethod.hpp>

#include <tmpc/Testing.hpp>


namespace tmpc :: testing
{
	/// @brief Test integration of a simple time-dependent ODE.
	class IrkSimpleTimeDependentOdeTest
	:	public Test
	{
	protected:
		using Real = double;
		static size_t constexpr NX = 1;
		static size_t constexpr NZ = 0;
		static size_t constexpr NU = 1;


		template <typename Integrator>
		void testIntegrate(Integrator const& integrator, double abs_tol, double rel_tol)
		{
			using VecX = blaze::StaticVector<Real, NX, blaze::columnVector>;
			using VecU = blaze::StaticVector<Real, NU, blaze::columnVector>;

			auto ode = [] (Real t, auto const& xdot, auto const& x, auto const& z,
				auto const& u, auto& f, auto& Jxdot, auto& Jx, auto& Jz)
			{
				f = u * t - xdot;
				Jxdot = {{-1.}};
				Jx = {{0.}};
			};

			Real const t0 = 0.11;
			VecX const x0 {3.};
			VecU const u {2.};
			Real const h = 0.05;
			VecX x1;
			integrator(ode, t0, h, x0, u, x1);

			TMPC_EXPECT_APPROX_EQ(x1, x0 + u * (pow(t0 + h, 2) - pow(t0, 2)) / 2., abs_tol, rel_tol);
		}
	};


	TEST_F(IrkSimpleTimeDependentOdeTest, testBackwardEulerDynamic)
	{
		testIntegrate(DynamicImplicitRungeKutta<Real> {BackwardEulerMethod {}, NX, NZ, NU}, 0., 0.001);
	}


	TEST_F(IrkSimpleTimeDependentOdeTest, testGaussLegendre2Dynamic)
	{
		testIntegrate(DynamicImplicitRungeKutta<Real> {GaussLegendreMethod {2}, NX, NZ, NU}, 0., 1e-14);
	}


	TEST_F(IrkSimpleTimeDependentOdeTest, testGaussLegendre3Dynamic)
	{
		testIntegrate(DynamicImplicitRungeKutta<Real> {GaussLegendreMethod {3}, NX, NZ, NU}, 0., 1e-17);
	}


	TEST_F(IrkSimpleTimeDependentOdeTest, testBackwardEulerStatic)
	{
		testIntegrate(StaticImplicitRungeKutta<Real, 1, NX, NZ, NU> {BackwardEulerMethod {}}, 0., 0.001);
	}


	TEST_F(IrkSimpleTimeDependentOdeTest, testGaussLegendre2Static)
	{
		testIntegrate(StaticImplicitRungeKutta<Real, 2, NX, NZ, NU> {GaussLegendreMethod {2}}, 0., 1e-14);
	}


	TEST_F(IrkSimpleTimeDependentOdeTest, testGaussLegendre3Static)
	{
		testIntegrate(StaticImplicitRungeKutta<Real, 3, NX, NZ, NU> {GaussLegendreMethod {3}}, 0., 1e-17);
	}
}