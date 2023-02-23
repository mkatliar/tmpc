#include <tmpc/integrator/ImplicitRungeKutta.hpp>
#include <tmpc/integrator/StaticImplicitRungeKutta.hpp>
#include <tmpc/integrator/BackwardEulerMethod.hpp>
#include <tmpc/integrator/GaussLegendreMethod.hpp>

#include "MassSpringDamperTest.hpp"


namespace tmpc :: testing
{
	//************************
	//
	// Mass-spring-damper test
	//
	//************************

	TEST_F(MassSpringDamperTest, testGaussLegendre2)
	{
		testIntegrate(
			ImplicitRungeKutta<Real> {GaussLegendreMethod {2}, NX, NZ, NU, NR}
		);
	}


	TEST_F(MassSpringDamperTest, testGaussLegendre2Sensitivities)
	{
		testIntegrateWithSensitivities(
			ImplicitRungeKutta<Real> {GaussLegendreMethod {2}, NX, NZ, NU, NR}
		);
	}


	TEST_F(MassSpringDamperTest, testGaussLegendre2LeastSquaresLagrangeTerm)
	{
		testIntegrateLeastSquaresLagrangeTerm(
			ImplicitRungeKutta<Real> {GaussLegendreMethod {2}, NX, NZ, NU, NR}
		);
	}


	TEST_F(MassSpringDamperTest, testStaticGaussLegendre2)
	{
		testIntegrate(
			StaticImplicitRungeKutta<Real, 2, NX, NZ, NU, NR> {GaussLegendreMethod {2}}
		);
	}


	TEST_F(MassSpringDamperTest, testStaticGaussLegendre2Sensitivities)
	{
		testIntegrateWithSensitivities(
			StaticImplicitRungeKutta<Real, 2, NX, NZ, NU, NR> {GaussLegendreMethod {2}}
		);
	}


	TEST_F(MassSpringDamperTest, testStaticGaussLegendre2LeastSquaresLagrangeTerm)
	{
		testIntegrateLeastSquaresLagrangeTerm(
			StaticImplicitRungeKutta<Real, 2, NX, NZ, NU, NR> {GaussLegendreMethod {2}}
		);
	}
}