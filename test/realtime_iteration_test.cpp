#include <tmpc/core/RealtimeIteration.hpp>
#include <tmpc/qp/CondensingSolver.hpp>
#include <tmpc/qp/HPMPCSolver.hpp>
#include <tmpc/integrator/rk4.hpp>
#include <tmpc/Matrix.hpp>
#include <tmpc/core/problem_specific.hpp>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <type_traits>

using namespace tmpc;

struct Dimensions
{
	static unsigned const NX = 2;
	static unsigned const NU = 1;
	static unsigned const NW = 0;
	static unsigned const NY = 0;
	static unsigned const NP = 0;
	static unsigned const NC = 0;
	static unsigned const NCT = 0;
};

typedef tmpc::ProblemSpecific<double, Dimensions> PS;

// An empty class for ODE, since we provide the discrete-time dynamics ourselves.
struct ODE
{
	ODE() {}
};

class SampleOCP
{
public:
	static unsigned const NX  = Dimensions::NX;
	static unsigned const NU  = Dimensions::NU;
	static unsigned const NC  = 0;
	static unsigned const NCT = 0;

	typedef StaticVector<double, NX> StateVector;
	typedef StaticVector<double, NU> InputVector;

	SampleOCP(std::size_t nt)
	:	nt_(nt)
	{
		A = {{1.,  1.},
			 {0.,  1.}};

		B = {{0.5},
			 {1. }};

		_x_min = {-1., -1.};
		_x_max = { 1.,  1.};
		_u_min = {-1.};
		_u_max = { 1.};
		_x_terminal_min = {-1., -1.};
		_x_terminal_max = { 1.,  1.};
	}

	// TODO: Move this property to RealtimeIteration?
	std::size_t getNumberOfIntervals() const
	{
		return nt_;
	}

	ODE const& getODE() const
	{
		static ODE const ode;
		return ode;
	}

	template <typename Stage>
	void InitStage(Stage& stage) const
	{
		StaticMatrix<double, NX, NX> Q {
			{66,  78},
			{78,  93}
		};

		StaticMatrix<double, NU, NU> R {
			{126}
		};

		StaticMatrix<double, NX, NU> S {
			{90},
			{108}
		};

		StaticVector<double, NX> q {0., 0.};

		StaticVector<double, NU> r {0.};

		stage.set_Q(Q);
		stage.set_R(R);
		stage.set_S(S);
		stage.set_q(q);
		stage.set_r(r);

		stage.set_A(A);
		stage.set_B(B);
		stage.set_x_next(A * stage.get_x() + B * stage.get_u());

		stage.set_x_min(_x_min);
		stage.set_x_max(_x_max);
		stage.set_u_min(_u_min);
		stage.set_u_max(_u_max);

		// No path constraints:
		//stage.set_C(...);
		//stage.set_D(...);
		//stage.set_d_min(...);
		//stage.set_d_max(...);
	}

	template <typename Stage>
	void InitTerminalStage(Stage& stage) const
	{
		StaticMatrix<double, NX, NX> H {
			{10, 14},
			{14, 20}
		};

		StaticVector<double, NX> g {0, 0};

		stage.set_Q(H);
		stage.set_q(g);

		stage.set_x_min(_x_terminal_min);
		stage.set_x_max(_x_terminal_max);

		// No terminal constraints:
		//stage.set_C(...);
		//stage.set_d_min(...);
		//stage.set_d_max(...);
	}

	template <typename Stage>
	void UpdateStage(Stage& stage) const
	{
		stage.set_x_next(A * stage.get_x() + B * stage.get_u());
	}

	template <typename Stage>
	void UpdateTerminalStage(Stage& stage) const
	{
		// Nothing to update.
	}

private:
	std::size_t nt_;
	StateVector _x_min;
	StateVector _x_max;
	StateVector _x_terminal_min;
	StateVector _x_terminal_max;
	InputVector _u_min;
	InputVector _u_max;

	StaticMatrix<double, NX, NX> A;
	StaticMatrix<double, NX, NU> B;
};

typedef SampleOCP OCP;

template <typename RealtimeIteration>
class RealtimeIterationTest : public ::testing::Test
{
public:
	RealtimeIterationTest(unsigned Nt = 2)
	:	_ocp(Nt)
	,	_qpSolver(Nt)
	,	_rti(_ocp, _qpSolver, WorkingPoint(Nt, OCP::StateVector(0.), OCP::InputVector(0.)))
	{
	}

protected:
	//typedef tmpc::RealtimeIteration<OCP, Integrator, QPSolver> RealtimeIteration;
	typedef typename RealtimeIteration::WorkingPoint WorkingPoint;
	typedef typename RealtimeIteration::QPSolver QPSolver;

	OCP _ocp;
	QPSolver _qpSolver;
	RealtimeIteration _rti;

	void Preparation()
	{
		_rti.Preparation();
	}

	OCP::InputVector Feedback(OCP::StateVector const& x0)
	{
		return _rti.Feedback(x0);
	}
};

typedef ::testing::Types<
		// TODO: make Solver a template template parameter of RealtimeIteration?
		tmpc::RealtimeIteration<K, Dimensions, OCP, tmpc::CondensingSolver<double, Dimensions>>
,		tmpc::RealtimeIteration<K, Dimensions, OCP, tmpc::HPMPCSolver     <double, Dimensions>>
	> RTITypes;

TYPED_TEST_CASE(RealtimeIterationTest, RTITypes);

TYPED_TEST(RealtimeIterationTest, DISABLED_GivesCorrectU0)
{
	PS::StateVector x;
	PS::InputVector u;

	// Step 0
	{
		this->Preparation();

		x = {1, 0};
		u = this->Feedback(x);

		PS::InputVector u_expected {-0.690877362606266};

		EXPECT_PRED2(MatrixApproxEquality(1e-6), u, u_expected);
	}

	// Step 1
	{
		this->Preparation();

		{
			PS::StateVector x;
			PS::InputVector u;
			MatrixApproxEquality is_approx(1e-6);

			x = {0.654561318696867,	 -0.690877362606266};	u = {0.215679569867116};
			EXPECT_PRED2(is_approx, this->_rti.getWorkingPoint().get_x(0), x);
			EXPECT_PRED2(is_approx, this->_rti.getWorkingPoint().get_u(0), u);

			x = {0.0715237410241597, -0.475197792739149};	u = {0.215679569867116};
			EXPECT_PRED2(is_approx, this->_rti.getWorkingPoint().get_x(1), x);
			EXPECT_PRED2(is_approx, this->_rti.getWorkingPoint().get_u(1), u);

			x = {0.0715237410241597, -0.475197792739149};
			EXPECT_PRED2(is_approx, this->_rti.getWorkingPoint().get_x(2), x);
		}

		x = {0.654561318696867,	-0.690877362606266};
		u = this->Feedback(x);

		PS::InputVector u_expected { 0.218183 };
		EXPECT_PRED2(MatrixApproxEquality(1e-5), u, u_expected);
	}
}

TEST(QpSizeTest, test_RtiQpSize)
{
	auto const nx = 5u;
	auto const nu = 1u;
	auto const nc = 3u;
	auto const nct = 4u;
	auto const nt = 2u;

	EXPECT_THAT(tmpc::RtiQpSize(nt, nx, nu, nc, nct),
			::testing::ElementsAre(tmpc::QpSize(nx, nu, nc), tmpc::QpSize(nx, nu, nc), tmpc::QpSize(nx, 0, nct)));
}
