/*
 * hpmpc_test.cpp
 *
 *  Created on: Jun 17, 2016
 *      Author: kotlyar
 */

#include "../include/qp/HPMPCSolver.hpp"
#include "../include/qp/Printing.hpp"

#include "qp_test_problems.hpp"

#include <gtest/gtest.h>

#include <iostream>

unsigned const NX = 2;
unsigned const NU = 1;
unsigned const NC = 0;
unsigned const NCT = 0;
unsigned const NT = 2;

typedef tmpc::HPMPCSolver<NX, NU, NC, NCT> Solver;
typedef Solver::Problem Problem;
typedef Solver::Solution Solution;

namespace
{
	std::ostream& operator<<(std::ostream& os, Solution const& point)
	{
		//typedef typename camels::CondensingSolver<NX_, NU_, NC_, NCT_>::size_type size_type;
		typedef unsigned size_type;
		for (size_type i = 0; i < point.nT(); ++i)
			os << point.get_x(i).transpose() << "\t" << point.get_u(i).transpose() << std::endl;

		return os << point.get_xend().transpose() << std::endl;
	}
}

TEST(hpmpc_test, problem_test)
{
	Problem qp(NT);
	set_zMin(qp, 0, -1.);	set_zMax(qp, 0, 1.);
	set_zMin(qp, 1, -1.);	set_zMax(qp, 1, 1.);
	set_zendMin(qp, -1.);	set_zendMax(qp, 1.);

	EXPECT_EQ(get_zMin(qp, 0), Problem::StateInputVector::Constant(-1.));
	EXPECT_EQ(get_zMax(qp, 0), Problem::StateInputVector::Constant( 1.));
	EXPECT_EQ(get_zMin(qp, 1), Problem::StateInputVector::Constant(-1.));
	EXPECT_EQ(get_zMax(qp, 1), Problem::StateInputVector::Constant( 1.));
	EXPECT_EQ(get_zendMin(qp), Problem::StateVector::Constant(-1.));
	EXPECT_EQ(get_zendMax(qp), Problem::StateVector::Constant( 1.));

	EXPECT_EQ(Eigen::Map<Problem::StateInputVector const>(qp.lb_data()[0]), Problem::StateInputVector::Constant(-1.));
	EXPECT_EQ(Eigen::Map<Problem::StateInputVector const>(qp.ub_data()[0]), Problem::StateInputVector::Constant( 1.));
	EXPECT_EQ(Eigen::Map<Problem::StateInputVector const>(qp.lb_data()[1]), Problem::StateInputVector::Constant(-1.));
	EXPECT_EQ(Eigen::Map<Problem::StateInputVector const>(qp.ub_data()[1]), Problem::StateInputVector::Constant( 1.));
	EXPECT_EQ(Eigen::Map<Problem::StateVector const>(qp.lb_data()[2]), Problem::StateVector::Constant(-1.));
	EXPECT_EQ(Eigen::Map<Problem::StateVector const>(qp.ub_data()[2]), Problem::StateVector::Constant( 1.));

	// Stage 0
	Problem::StageHessianMatrix H0;
	H0 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
	H0 = H0.transpose() * H0;	// Make positive definite.

	const Eigen::MatrixXd Q0 = H0.topLeftCorner(qp.nX(), qp.nX());
	const Eigen::MatrixXd R0 = H0.bottomRightCorner(qp.nU(), qp.nU());
	const Eigen::MatrixXd S0 = H0.topRightCorner(qp.nX(), qp.nU());
	const Eigen::MatrixXd S0T = H0.bottomLeftCorner(qp.nU(), qp.nX());

	Eigen::MatrixXd A0(qp.nX(), qp.nX());
	A0 << 1, 1, 0, 1;

	Eigen::MatrixXd B0(qp.nX(), qp.nU());
	B0 << 0.5, 1.0;

	Eigen::VectorXd a0(qp.nX());
	a0 << 1, 2;

	// Stage 1
	Problem::StageHessianMatrix H1;
	H1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
	H1 = H1.transpose() * H1;	// Make positive definite.

	const Eigen::MatrixXd Q1 = H1.topLeftCorner(qp.nX(), qp.nX());
	const Eigen::MatrixXd R1 = H1.bottomRightCorner(qp.nU(), qp.nU());
	const Eigen::MatrixXd S1 = H1.topRightCorner(qp.nX(), qp.nU());
	const Eigen::MatrixXd S1T = H1.bottomLeftCorner(qp.nU(), qp.nX());

	Eigen::MatrixXd A1(qp.nX(), qp.nX());
	A1 << 1, 1, 0, 1;

	Eigen::MatrixXd B1(qp.nX(), qp.nU());
	B1 << 0.5, 1.0;

	Eigen::VectorXd a1(qp.nX());
	a1 << 1, 2;

	// Stage 2
	Eigen::MatrixXd H2(qp.nX(), qp.nX());
	H2 << 1, 2, 3, 4;
	H2 = H2.transpose() * H2;	// Make positive definite.

	const Eigen::MatrixXd Q2 = H2.topLeftCorner(qp.nX(), qp.nX());

	// Setup QP
	set_H(qp, 0, H0);
	EXPECT_EQ(get_H(qp, 0), H0);
	EXPECT_EQ(Eigen::Map<Problem::HPMPC_QMatrix const>(qp.Q_data()[0]), Q0);
	EXPECT_EQ(Eigen::Map<Problem::HPMPC_RMatrix const>(qp.R_data()[0]), R0);
	EXPECT_EQ(Eigen::Map<Problem::HPMPC_SMatrix const>(qp.S_data()[0]), S0.transpose());

	set_H(qp, 1, H1);
	EXPECT_EQ(get_H(qp, 1), H1);
	EXPECT_EQ(Eigen::Map<Problem::HPMPC_QMatrix const>(qp.Q_data()[1]), Q1);
	EXPECT_EQ(Eigen::Map<Problem::HPMPC_RMatrix const>(qp.R_data()[1]), R1);
	EXPECT_EQ(Eigen::Map<Problem::HPMPC_SMatrix const>(qp.S_data()[1]), S1.transpose());

	set_Hend(qp, H2);
	EXPECT_EQ(get_Hend(qp), H2);
	EXPECT_EQ(Eigen::Map<Problem::HPMPC_QMatrix const>(qp.Q_data()[2]), Q2);
	EXPECT_EQ(qp.R_data()[2], nullptr);
	EXPECT_EQ(qp.S_data()[2], nullptr);

	EXPECT_NE(qp.q_data()[2], nullptr);
	EXPECT_EQ(qp.r_data()[2], nullptr);

	Problem::InterStageMatrix C0;
	C0 << A0, B0;
	set_C(qp, 0, C0);
	EXPECT_EQ(get_C(qp, std::size_t(0)), C0);
	EXPECT_EQ(Eigen::Map<Problem::HPMPC_AMatrix const>(qp.A_data()[0]), A0);
	EXPECT_EQ(Eigen::Map<Problem::HPMPC_BMatrix const>(qp.B_data()[0]), B0);

	set_c(qp, 0, a0);
	EXPECT_EQ(get_c(qp, 0), a0);
	EXPECT_EQ(Eigen::Map<Problem::StateVector const>(qp.b_data()[0]), a0);

	Problem::InterStageMatrix C1;
	C1 << A1, B1;
	set_C(qp, 1, C1);
	EXPECT_EQ(get_C(qp, 1), C1);
	EXPECT_EQ(Eigen::Map<Problem::HPMPC_AMatrix const>(qp.A_data()[1]), A1);
	EXPECT_EQ(Eigen::Map<Problem::HPMPC_BMatrix const>(qp.B_data()[1]), B1);

	set_c(qp, 1, a1);
	EXPECT_EQ(get_c(qp, 1), a1);
	EXPECT_EQ(Eigen::Map<Problem::StateVector const>(qp.b_data()[1]), a1);
 }

TEST(hpmpc_test, solve_test)
{
	Problem qp(NT);
	tmpc_test::qp_problems::problem_1(qp);

	Print_MATLAB(std::cout, qp, "qp");

	Solver solver(qp.nT());
	Solution solution(NT);

	try
	{
		solver.Solve(qp, solution);
	}
	catch (std::runtime_error const& e)
	{
		std::cout << "HPMPC SOLVER RETURNED AN ERROR" << std::endl;
		std::cout << "-- sol (multistage) --" << std::endl << solution << std::endl;
	}

	Solution::StateInputVector z0_expected;
	z0_expected << 1., -1., -1;
	EXPECT_TRUE(get_z(solution, 0).isApprox(z0_expected));

	Solution::StateInputVector z1_expected;
	z1_expected << 0.5, 0., -1;
	EXPECT_TRUE(get_z(solution, 1).isApprox(z1_expected));

	Solution::StateVector z2_expected;
	z2_expected << 1., 1;
	EXPECT_TRUE(get_xend(solution).isApprox(z2_expected));

	std::cout << "-- sol (multistage) --" << std::endl << solution << std::endl;
}

TEST(hpmpc_test, low_level_call_test)
{
	int const nx[NT + 1] = {NX, NX, NX};
	int const nu[NT + 1] = {NU, NU, 0};
	int const nb[NT + 1] = {NU + NX, NU + NX, NX};
	int const ng[NT + 1] = {0, 0, 0};

	double const A0[NX * NX] = {1., 1., 0., -1.};
	double const * const A[NT] = {A0, A0};

	double const B0[NX * NU] = {0.5, 1.};
	double const * const B[NT] = {B0, B0};

	double const b0[NX] = {1., 2.};
	double const * const b[NT] = {b0, b0};

	double const Q0[NX * NX] = {66., 78., 78., 93.};
	double const QT[NX * NX] = {10., 14., 14., 20.};
	double const * const Q[NT + 1] = {Q0, Q0, QT};

	double const S0[NU * NX] = {90., 108};
	double const * const S[NT + 1] = {S0, S0, nullptr};

	double const R0[NU * NU] = {126.};
	double const * const R[NT + 1] = {R0, R0, nullptr};

	double const q0[NX] = {0., 0.};
	double const * const q[NT + 1] = {q0, q0, q0};

	double const r0[NU] = {0.};
	double const * const r[NT + 1] = {r0, r0, nullptr};

	double const lb0[NU + NX] = {-1., -1., -1.};
	double const lbT[NX] = {-1., -1.};
	double const * const lb[NT + 1] = {lb0, lb0, lbT};

	double const ub0[NU + NX] = {1., 1., 1.};
	double const ubT[NX] = {1., 1.};
	double const * const ub[NT + 1] = {ub0, ub0, ubT};

	double const * const C[NT + 1] = {nullptr, nullptr, nullptr};
	double const * const D[NT + 1] = {nullptr, nullptr, nullptr};
	double const * const lg[NT + 1] = {nullptr, nullptr, nullptr};
	double const * const ug[NT + 1] = {nullptr, nullptr, nullptr};

	double x  [NT + 1][NX];	double * px [NT + 1] = {x [0], x [1], x [2]};
	double u  [NT + 1][NU]; double * pu [NT + 1] = {u [0], u [1], u [2]};
	double pi [NT    ][NX]; double * ppi[NT    ] = {pi[0], pi[1]};
	double lam[NT + 1][2 * (NX + NU)];	double * plam[NT + 1] = {lam[0], lam[1], lam[2]};
	double t  [NT + 1][2 * (NX + NU)];	double * pt  [NT + 1] = {t  [0], t  [1], t  [2]};
	double inf_norm_res[4];

	int const max_iter = 100;
	double stat[5 * max_iter];

	std::vector<char> workspace(hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(
			static_cast<int>(NT), const_cast<int*>(nx), const_cast<int*>(nu), const_cast<int*>(nb), const_cast<int*>(ng)));

	int num_iter = 0;
	double const mu0 = 0;
	double const mu_tol = 1e-10;

	auto const ret = c_order_d_ip_ocp_hard_tv(&num_iter, max_iter, mu0, mu_tol, NT,
			nx, nu, nb, ng, 1, A, B, b,
			Q, S, R, q, r, lb, ub, C, D,
			lg, ug, px, pu, ppi, plam, pt, inf_norm_res,
			static_cast<void *>(workspace.data()), stat);

	ASSERT_EQ(ret, 0);
}

TEST(Eigen, DISABLED_comma_initializer_test)
{
	auto const M = 0;
	auto const N1 = 2;
	auto const N2 = 1;

	Eigen::Matrix<double, M, N1> A1;
	Eigen::Matrix<double, M, N2> A2;
	Eigen::Matrix<double, M, N1 + N2> B;

	B << A1, A2;
}
