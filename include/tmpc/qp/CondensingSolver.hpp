#pragma once

#include "QuadraticProblem.hpp"
#include "QpOasesProblem.hpp"
#include "QpOasesSolver.hpp"

#include <qpOASES.hpp>
#include "Condensing.hpp"
#include "MultiStageQPSolution.hpp"
#include "UnsolvedQpException.hpp"

#include <ostream>

namespace tmpc {

namespace detail {
qpOASES::Options qpOASES_DefaultOptions();
}

/**
 * \brief Condensing solver using qpOASES
 *
 * \tparam <Scalar> Scalar type
 * \tparam <D> Class defining problem dimensions
 */
template <typename Scalar, typename D>
class CondensingSolver
{
	static auto constexpr NX = D::NX;
	static auto constexpr NU = D::NU;
	static auto constexpr NZ = NX + NU;
	static auto constexpr NC = D::NC;
	static auto constexpr NCT = D::NCT;

public:
	typedef QpOasesProblem CondensedQP;

	// Problem type for CondensingSolver
	// TODO: change MultiStageQuadraticProblem so that it takes D as a template parameter?
	typedef QuadraticProblem<double> Problem;

	// Solution data type
	// TODO: change MultiStageQPSolution so that it takes D as a template parameter?
	typedef MultiStageQPSolution<NX, NU, NC, NCT> Solution;

	// Exception that can be thrown from the Solve() member function.
	class SolveException;

	typedef std::size_t size_type;
	typedef DynamicVector<Scalar> Vector;

	CondensingSolver(size_type nt, qpOASES::Options const& options = detail::qpOASES_DefaultOptions())
	:	_Nt(nt)
	,	_condensedQP({condensedQpSize(RtiQpSize(nt, NX, NU, NC, NCT))})
	,	_condensedSolution(nIndep(nt))
	,	_problem(nIndep(nt), nDep(nt) + nConstr(nt))
	{
		_problem.setOptions(options);
	}

	/**
	 * \brief Copy constructor.
	 *
	 * Copying is not allowed.
	 */
	CondensingSolver(CondensingSolver const&) = delete;

	/**
	 * \brief Move constructor.
	 *
	 * Move-construction is ok.
	 */
	CondensingSolver(CondensingSolver&& rhs) = default;

	CondensingSolver& operator=(CondensingSolver const&) = delete;
	CondensingSolver& operator=(CondensingSolver&&) = delete;

	size_type nT() const { return _Nt; }
	size_type constexpr nX() { return NX; }
	size_type constexpr nZ() { return NZ; }
	size_type constexpr nU() { return NU; }
	size_type constexpr nD() { return NC; }
	size_type constexpr nDT() {	return NCT;	}
	size_type nIndep() const { return nIndep(nT()); }
	static size_type nIndep(size_type nt) { return NX + NU * nt; }
	size_type nDep() const { return nDep(nT()); }
	static size_type nDep(size_type nt) { return NX * nt; }
	size_type nVar() const { return nVar(nT()); }
	static size_type nVar(size_type nt) { return NZ * nt + NX; }
	static size_type nConstr(size_type nt) { return NC * nt + NCT; }

	const Vector& getCondensedSolution() const { return _condensedSolution;	}

	const CondensedQP& getCondensedQP() const noexcept { return _condensedQP; }
	bool getHotStart() const noexcept { return _hotStart; }

	// qpOASES-specific part
	//

	// Get maximum number of working set recalculations for qpOASES
	unsigned const getMaxWorkingSetRecalculations() const noexcept { return _maxWorkingSetRecalculations; }

	// Set maximum number of working set recalculations for qpOASES
	void setMaxWorkingSetRecalculations(unsigned val) noexcept { _maxWorkingSetRecalculations = val; }

private:
	// Number of time steps
	size_type _Nt;

	// Number of constraints per stage = nX() + nD().
	size_type nC() const { return nX() + nD(); }

	// Input data for qpOASES
	CondensedQP _condensedQP;

	// Output data from qpOASES
	Vector _condensedSolution;

	bool _hotStart = false;

	// TODO: wrap _problem into a pimpl to
	// a) Reduce dependencies
	// b) Avoid deep-copy of qpOASES::SQProblem object of move-construction of CondensingSolver.
	qpOASES::SQProblem _problem;
	unsigned _maxWorkingSetRecalculations = 1000;

public:
	void Solve(Problem const& msqp, Solution& solution)
	{
		// Check argument sizes.
		if (msqp.size() != nT())
			throw std::invalid_argument("CondensingSolver::Solve(): size of MultistageQP does not match solver sizes, sorry.");

		if (solution.nT() != nT())
			throw std::invalid_argument("CondensingSolver::Solve(): size of solution Point does not match solver sizes, sorry.");

		// Make a condensed problem.
		_condensedQP.front() = condense<double>(msqp.begin(), msqp.end());

		/* Solve the condensed QP. */
		int nWSR = static_cast<int>(_maxWorkingSetRecalculations);
		const auto res = _hotStart ?
			_problem.hotstart(_condensedQP.H_data(), _condensedQP.g_data(), _condensedQP.A_data(),
					_condensedQP.lb_data(), _condensedQP.ub_data(), _condensedQP.lbA_data(), _condensedQP.ubA_data(), nWSR) :
			_problem.init    (_condensedQP.H_data(), _condensedQP.g_data(), _condensedQP.A_data(),
					_condensedQP.lb_data(), _condensedQP.ub_data(), _condensedQP.lbA_data(), _condensedQP.ubA_data(), nWSR);

		if (res != qpOASES::SUCCESSFUL_RETURN)
			throw QpOasesSolveException(res, _condensedQP);

		solution.setNumIter(nWSR);
		_hotStart = true;

		/* Get solution of the condensed QP. */
		_problem.getPrimalSolution(_condensedSolution.data());
		//problem.getDualSolution(yOpt);

		// Calculate the solution of the multi-stage QP.
		// TODO: add function recoverSolution() to Condensing.hpp
		solution.set_x(0, subvector(_condensedSolution, 0, NX));
		for (size_type i = 0; i < nT(); ++i)
		{
			solution.set_u(i, subvector(_condensedSolution, NX + i * NU, NU));
			solution.set_x(i + 1, msqp[i].get_A() * solution.get_x(i) + msqp[i].get_B() * solution.get_u(i) + msqp[i].get_b());
		}
	}
};

}	// namespace tmpc
