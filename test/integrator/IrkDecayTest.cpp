#include <tmpc/integrator/ImplicitRungeKutta.hpp>
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

	TEST_F(DecayTest, testGaussLegendre2)
	{
		testIntegrate(
			ImplicitRungeKutta<Real> {GaussLegendreMethod {2}, NX, NZ, NU, NR}
		);
	}


	TEST_F(DecayTest, testGaussLegendre2Sensitivities)
	{
		testIntegrateWithSensitivities(
			ImplicitRungeKutta<Real> {GaussLegendreMethod {2}, NX, NZ, NU, NR}
		);
	}


	TEST_F(DecayTest, testGaussLegendre2LeastSquaresLagrangeTerm)
	{
		testIntegrateLeastSquaresLagrangeTerm(
			ImplicitRungeKutta<Real> {GaussLegendreMethod {2}, NX, NZ, NU, NR}
		);
	}


	TEST_F(DecayTest, testStaticGaussLegendre2)
	{
		testIntegrate(
			StaticImplicitRungeKutta<Real, 2, NX, NZ, NU, NR> {GaussLegendreMethod {2}}
		);
	}


	TEST_F(DecayTest, testStaticGaussLegendre2Sensitivities)
	{
		testIntegrateWithSensitivities(
			StaticImplicitRungeKutta<Real, 2, NX, NZ, NU, NR> {GaussLegendreMethod {2}}
		);
	}


	TEST_F(DecayTest, testStaticGaussLegendre2LeastSquaresLagrangeTerm)
	{
		testIntegrateLeastSquaresLagrangeTerm(
			StaticImplicitRungeKutta<Real, 2, NX, NZ, NU, NR> {GaussLegendreMethod {2}}
		);
	}
}
