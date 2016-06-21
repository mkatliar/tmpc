#pragma once

#include "HPMPCProblem.hpp"
#include "HPMPCSolution.hpp"

#include <c_interface.h>	// TODO: "c_interface.h" is a very general name; change the design so that it is <hpmpc/c_interface.h>, for example.


namespace tmpc
{
	void throw_hpmpc_error(int err_code);

	template<unsigned NX_, unsigned NU_, unsigned NC_, unsigned NCT_>
	class HPMPCSolver
	{
	public:
		static unsigned const NX = NX_;
		static unsigned const NU = NU_;
		static unsigned const NZ = NX + NU;
		static unsigned const NC = NC_;
		static unsigned const NCT = NCT_;

		typedef HPMPCProblem<NX, NU, NC, NCT> Problem;
		typedef HPMPCSolution<NX, NU, NC, NCT> Solution;

		HPMPCSolver(std::size_t nt, int max_iter = 100)
		:	_nx(nt + 1, NX)
		,	_nu(nt + 1, NU)
		,	_nb(nt + 1, NU + NX)
		,	_ng(nt + 1, NC)
		,	_stat(max_iter)
		{
			_nu.back() = 0;
			_nb.back() = NX;
			_ng.back() = NCT;

			// Allocate workspace
			_workspace.resize(hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(
					static_cast<int>(nt), _nx.data(), _nu.data(), _nb.data(), _ng.data()));
		}

		void Solve(Problem const& p, Solution& s)
		{
			int num_iter = 0;
			auto const ret = c_order_d_ip_ocp_hard_tv(&num_iter, getMaxIter(), _mu0, _muTol, nT(),
					_nx.data(), _nu.data(), _nb.data(), _ng.data(), _warmStart ? 1 : 0, p.A_data(), p.B_data(), p.b_data(),
					p.Q_data(), p.S_data(), p.R_data(), p.q_data(), p.r_data(), p.lb_data(), p.ub_data(), p.C_data(), p.D_data(),
					p.lg_data(), p.ug_data(), s.x_data(), s.u_data(), s.pi_data(), s.lam_data(), s.t_data(), s.inf_norm_res_data(),
					static_cast<void *>(_workspace.data()), _stat[0].data());

			if (ret != 0)
				throw_hpmpc_error(ret);

			_warmStart = true;
		}

		std::size_t nT() const noexcept { return _nx.size() - 1; }
		std::size_t getMaxIter() const noexcept { return _stat.size(); }

	private:
		// Array of NX sizes
		std::vector<int> _nx;

		// Array of NU sizes
		std::vector<int> _nu;

		// Array of NB (bound constraints) sizes
		std::vector<int> _nb;

		// Array of NG (path constraints) sizes
		std::vector<int> _ng;

		// Workspace for HPMPC functions
		std::vector<char> _workspace;

		// Iteration statistics. HPMPC returns 5 double numbers per iteration.
		typedef std::array<double, 5> IterStat;
		std::vector<IterStat> _stat;

		double _mu0 = 0.;
		double _muTol = 1e-10;
		bool _warmStart = false;
	};
}
