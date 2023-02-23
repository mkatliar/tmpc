#include <tmpc/integrator/DynamicImplicitRungeKutta.hpp>
#include <tmpc/integrator/StaticImplicitRungeKutta.hpp>
#include <tmpc/integrator/BackwardEulerMethod.hpp>
#include <tmpc/integrator/GaussLegendreMethod.hpp>

#include "DecayTest.hpp"


namespace tmpc :: testing
{
	//************************
	//
	// Exponential decay test
	//
	//************************

	TEST_F(DecayTest, testGaussLegendre2Dynamic)
	{
		testIntegrate(
			DynamicImplicitRungeKutta<Real> {GaussLegendreMethod {2}, NX, NZ, NU, NR}
		);
	}


	TEST_F(DecayTest, testGaussLegendre2SensitivitiesDynamic)
	{
		testIntegrateWithSensitivities(
			DynamicImplicitRungeKutta<Real> {GaussLegendreMethod {2}, NX, NZ, NU, NR}
		);
	}


	TEST_F(DecayTest, testGaussLegendre2LeastSquaresLagrangeTermDynamic)
	{
		testIntegrateLeastSquaresLagrangeTerm(
			DynamicImplicitRungeKutta<Real> {GaussLegendreMethod {2}, NX, NZ, NU, NR}
		);
	}


	TEST_F(DecayTest, testGaussLegendre2Static)
	{
		testIntegrate(
			StaticImplicitRungeKutta<Real, 2, NX, NZ, NU, NR> {GaussLegendreMethod {2}}
		);
	}


	TEST_F(DecayTest, testGaussLegendre2SensitivitiesStatic)
	{
		testIntegrateWithSensitivities(
			StaticImplicitRungeKutta<Real, 2, NX, NZ, NU, NR> {GaussLegendreMethod {2}}
		);
	}


	TEST_F(DecayTest, testGaussLegendre2LeastSquaresLagrangeTermStatic)
	{
		testIntegrateLeastSquaresLagrangeTerm(
			StaticImplicitRungeKutta<Real, 2, NX, NZ, NU, NR> {GaussLegendreMethod {2}}
		);
	}
}
