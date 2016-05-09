#pragma once

#include <ostream>
#include <functional>
#include <stdexcept>

namespace camels
{
	template<class _Problem, typename Integrator_, class QPSolver_>
	class ModelPredictiveController
	{
	public:
		typedef _Problem Problem;
		typedef QPSolver_ QPSolver;
		typedef Integrator_ Integrator;

		typedef typename QPSolver::Point Trajectory;

		typedef typename Problem::StateVector StateVector;
		typedef typename Problem::InputVector InputVector;
		typedef typename Problem::StateInputVector StateInputVector;
		typedef typename Problem::ODEJacobianMatrix ODEJacobianMatrix;
		typedef typename Problem::LagrangeHessianMatrix LagrangeHessianMatrix;
		typedef typename Problem::MayerHessianMatrix MayerHessianMatrix;

		typedef std::function<void (typename QPSolver::MultiStageQP const&)> QPCallback;

		ModelPredictiveController(Problem const& ocp, Integrator const& integrator, Trajectory const& working_point)
		:	_ocp(ocp)
		,	_QP(working_point.nT())
		,	_workingPoint(working_point)
		,	_solution(working_point.nT())
		,	_Solver(working_point.nT())
		,	_levenbergMarquardt(0.0)
		,	_integrator(integrator)
		,	_prepared(false)
		{
		}

		// Feed current state x0, get back control input u.
		InputVector Feedback(const StateVector& x0)
		{
			if (!_prepared)
				throw std::logic_error("ModelPredictiveController::Feedback(): controller is not prepared.");

			/** embed current initial value */
			_QP.xMin(0) = _QP.xMax(0) = x0 - _workingPoint.x(0);

			// Call the QP callback, if there is one.
			if(_QPCallback)
				_QPCallback(_QP);

			/** solve QP */
			_Solver.Solve(_QP, _solution);

			_prepared = false;

			// Return the calculated control input.
			return _workingPoint.u(0) + _solution.u(0);
		}

		void Preparation()
		{
			if (_prepared)
				throw std::logic_error("ModelPredictiveController::Preparation(): controller is already prepared.");

			// Add QP step to the working point.
			_workingPoint += _solution;

			/** prepare QP for next solution */
			//qpDUNES_shiftLambda(&_qpData);			/* shift multipliers */
			//qpDUNES_shiftIntervals(&_qpData);		/* shift intervals (particularly important when using qpOASES for underlying local QPs) */

			// Shift working point
			shift(_workingPoint);

			// Calculate new QP.
			UpdateQP();

			_prepared = true;
		}

		void PrintQP_C(std::ostream& log_stream) const
		{
			_QP.PrintQP_C(log_stream);
		}

		void PrintQP_MATLAB(std::ostream& log_stream) const
		{
			_QP.PrintQP_MATLAB(log_stream);
		}

		void PrintQP_zMax_C(std::ostream& log_stream) const;
		void PrintQP_zMin_C(std::ostream& log_stream) const;

		// Log working point
		void PrintWorkingPoint_MATLAB(std::ostream& os, const std::string& var_name) const
		{
			for (unsigned i = 0; i < nT(); ++i)
				os << var_name << "{" << i + 1 << "} = [" << _workingPoint.w(i) << "];" << std::endl;
			os << var_name << "{" << nT() + 1 << "} = [" << _workingPoint.wend() << "];" << std::endl;
		}
		
		double getLevenbergMarquardt() const { return _levenbergMarquardt; }
		void setLevenbergMarquardt(double val) { _levenbergMarquardt = val; }

		unsigned nT() const { return _ocp.getNumberOfIntervals(); }
		unsigned nU() const	noexcept { return _Nu; }
		unsigned nX() const	noexcept { return _Nx; }
		unsigned nZ() const noexcept { return _Nz; }

		void setQPCallback(const QPCallback& cb) { _QPCallback = cb; }

	private:

		// Initializes _G, _g, _y, _C, _c, _zMin, _zMax based on current working point _w.
		void UpdateQP()
		{
			for (unsigned i = 0; i < _ocp.getNumberOfIntervals(); ++i)
				UpdateStage(i);

			// End state constraints.
			typename Problem::TerminalConstraintJacobianMatrix D;
			typename Problem::TerminalConstraintVector d_min, d_max;
			_ocp.TerminalConstraints(_workingPoint.wend(), D, d_min, d_max);
			_QP.Dend() = D;
			_QP.dendMin() = d_min;
			_QP.dendMax() = d_max;

			_QP.zendMin() = _ocp.getTerminalStateMin() - _workingPoint.wend();
			_QP.zendMax() = _ocp.getTerminalStateMax() - _workingPoint.wend();

			// Hessian and gradient of Mayer term.
			typename Problem::MayerHessianMatrix H_T;
			typename Problem::StateVector g_T;
			_ocp.MayerTerm(_workingPoint.wend(), g_T, H_T);

			// Adding Levenberg-Marquardt term to make H positive-definite.
			_QP.Hend() = H_T + _levenbergMarquardt * MayerHessianMatrix::Identity();
			_QP.gend() = g_T;
		}

		void UpdateStage(unsigned i)
		{
			// Hessians and gradients of Lagrange terms.
			//
			LagrangeHessianMatrix H_i;
			StateInputVector g_i;

			_ocp.LagrangeTerm(i, _workingPoint.w(i), g_i, H_i);

			// Adding Levenberg-Marquardt term to make H positive-definite.
			_QP.H(i) = H_i + _levenbergMarquardt * LagrangeHessianMatrix::Identity();
			_QP.g(i) = g_i;

			// Bound constraints.
			StateInputVector z_min, z_max;
			z_min << _ocp.getStateMin(), _ocp.getInputMin();
			z_max << _ocp.getStateMax(), _ocp.getInputMax();

			// C = [ssA, ssB];
			// x_{k+1} = C * z_k + c_k
			typename Problem::StateVector x_plus;
			typename Problem::ODEJacobianMatrix J;
			_integrator.Integrate(i * _integrator.timeStep(), _workingPoint.w(i), x_plus, J);
			_QP.C(i) = J;

			// \Delta x_{k+1} = C \Delta z_k + f(z_k) - x_{k+1}
			// c = f(z_k) - x_{k+1}
			_QP.c(i) = x_plus - _workingPoint.x(i + 1);

			typename Problem::ConstraintJacobianMatrix D;
			typename Problem::ConstraintVector d_min, d_max;
			_ocp.PathConstraints(i, _workingPoint.w(i), D, d_min, d_max);
			_QP.D(i) = D;
			_QP.dMin(i) = d_min;
			_QP.dMax(i) = d_max;

			// z_min stores _Nt vectors of size _Nz and 1 vector of size _Nx
			_QP.zMin(i) = z_min - _workingPoint.w(i);

			// z_max stores _Nt vectors of size _Nz and 1 vector of size _Nx
			_QP.zMax(i) = z_max - _workingPoint.w(i);
		}

		// Private data members.
		Problem const& _ocp;
		Integrator const& _integrator;

		static const unsigned _Nu = Problem::NU;
		static const unsigned _Nx = Problem::NX;
		static const unsigned _Nz = Problem::NW;
		static const unsigned _Nd = Problem::NC;
		static const unsigned _NdT = Problem::NCT;
		
		typename QPSolver::MultiStageQP _QP;
		typename QPSolver::Point _solution;
		QPSolver _Solver;

		// A callback to call back before solving each QP.
		QPCallback _QPCallback;

		double _levenbergMarquardt;

		// Working point (linearization point).
		Trajectory _workingPoint;

		// Preparation() sets this flag to true, Feedback() resets it to false.
		bool _prepared;
	};
}