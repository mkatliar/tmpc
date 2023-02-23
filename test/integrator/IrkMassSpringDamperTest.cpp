#include <tmpc/integrator/DynamicImplicitRungeKutta.hpp>
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

	TEST_F(MassSpringDamperTest, testGaussLegendre2Dynamic)
	{
		testIntegrate(
			DynamicImplicitRungeKutta<Real> {GaussLegendreMethod {2}, NX, NZ, NU, NR}
		);
	}


	TEST_F(MassSpringDamperTest, testGaussLegendre2SensitivitiesDynamic)
	{
		testIntegrateWithSensitivities(
			DynamicImplicitRungeKutta<Real> {GaussLegendreMethod {2}, NX, NZ, NU, NR}
		);
	}


	TEST_F(MassSpringDamperTest, testGaussLegendre2LeastSquaresLagrangeTermDynamic)
	{
		testIntegrateLeastSquaresLagrangeTerm(
			DynamicImplicitRungeKutta<Real> {GaussLegendreMethod {2}, NX, NZ, NU, NR}
		);
	}


	TEST_F(MassSpringDamperTest, testGaussLegendre2Static)
	{
		testIntegrate(
			StaticImplicitRungeKutta<Real, 2, NX, NZ, NU, NR> {GaussLegendreMethod {2}}
		);
	}


	TEST_F(MassSpringDamperTest, testGaussLegendre2SensitivitiesStatic)
	{
		testIntegrateWithSensitivities(
			StaticImplicitRungeKutta<Real, 2, NX, NZ, NU, NR> {GaussLegendreMethod {2}}
		);
	}


	TEST_F(MassSpringDamperTest, testGaussLegendre2LeastSquaresLagrangeTermStatic)
	{
		testIntegrateLeastSquaresLagrangeTerm(
			StaticImplicitRungeKutta<Real, 2, NX, NZ, NU, NR> {GaussLegendreMethod {2}}
		);
	}
}