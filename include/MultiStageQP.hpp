#pragma once

#include "MultiStageQPSize.hpp"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <vector>
#include <memory>
#include <ostream>
#include <cstdlib>

namespace camels
{
	/* MultiStageQP represents a problem
	*
	*	min  sum_{ k = 0..nI } z_k'*H_k*z_k + g_k'*z_k
	*	s.t. x_{ k + 1 } = C_k * z_k + c_k				for k = 0..nI - 1
	*            dLow_k <= D_k * z_k <= dUpp_k			for k = 0..nI
	*            zMin_k <= z_k <= zMax_k                for k = 0..nI
	*
	*	where x_k is implicitly defined by z_k = [x_k  u_k] as the first nX variables of z_k
	*
	*	It holds
	*	z_k  \in R^nZ  for k = 0..nI - 1
	*   z_nI \in R*nX
	*
	*	nX < nZ
	*	nU = nZ - nX
	*/
	//template<unsigned NX_, unsigned NU_, unsigned NC_, unsigned NCT_>
	class MultiStageQP
	{
	public:
		typedef unsigned int size_type;

		/*
		static unsigned const NX = NX_;
		static unsigned const NU = NU_;
		static unsigned const NC = NC_;
		static unsigned const NCT = NCT_;
		*/

		typedef Eigen::Map<Eigen::VectorXd> VectorMap;
		typedef Eigen::Map<const Eigen::VectorXd> VectorConstMap;
		typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix;
		typedef Eigen::Map<RowMajorMatrix> RowMajorMatrixMap;
		typedef Eigen::Map<const RowMajorMatrix> RowMajorMatrixConstMap;

		MultiStageQP(size_type nx, size_type nu, size_type nd, size_type ndt, size_type nt)
		:	MultiStageQP(MultiStageQPSize(nx, nu, nd, ndt, nt))
		{
		}

		void PrintQP_C(std::ostream& os) const;

		void PrintQP_zMax_C(std::ostream& log_stream) const;

		void PrintQP_zMin_C(std::ostream& log_stream) const;

		void PrintQP_MATLAB(std::ostream& log_stream, const std::string& var_name = "qp") const;

		const MultiStageQPSize& size() const;

		size_type nT() const { return _size.nT(); }
		size_type nX() const { return _size.nX(); }
		size_type nZ() const { return _size.nZ(); }
		size_type nU() const { return _size.nU(); }
		size_type nD() const { return _size.nD(); }
		size_type nDT() const { return _size.nDT(); }

		size_type nIndep() const { return _size.nIndep(); }
		size_type nDep() const { return _size.nDep(); }
		size_type nVar() const { return _size.nVar(); }
		size_type nConstr() const { return _size.nConstr(); }

		RowMajorMatrixMap H(unsigned i)
		{
			assert(i < nT() + 1);
			const auto sz = i < nT() ? nZ() : nX();
			return RowMajorMatrixMap(_H.data() + i * nZ() * nZ(), sz, sz);
		}

		RowMajorMatrixConstMap H(unsigned i) const
		{
			assert(i < nT() + 1);
			const auto sz = i < nT() ? nZ() : nX();
			return RowMajorMatrixConstMap(_H.data() + i * nZ() * nZ(), sz, sz);
		}

		VectorMap g(unsigned i)
		{
			assert(i < nT() + 1);
			return VectorMap(_g.data() + i * nZ(), i < nT() ? nZ() : nX());
		}

		VectorConstMap g(unsigned i) const
		{
			assert(i < nT() + 1);
			return VectorConstMap(_g.data() + i * nZ(), i < nT() ? nZ() : nX());
		}
		
		RowMajorMatrixMap C(unsigned i)
		{
			return RowMajorMatrixMap(_C.data() + i * nX() * nZ(), nX(), nZ());
		}

		RowMajorMatrixConstMap C(unsigned i) const
		{
			return RowMajorMatrixConstMap(_C.data() + i * nX() * nZ(), nX(), nZ());
		}

		RowMajorMatrixMap D(unsigned i)
		{
			assert(i <= nT());
			return RowMajorMatrixMap(_D.data() + i * nD() * nZ(), i < nT() ? nD() : nDT(), i < nT() ? nZ() : nX());
		}

		RowMajorMatrixConstMap D(unsigned i) const
		{
			assert(i <= nT());
			return RowMajorMatrixConstMap(_D.data() + i * nD() * nZ(), i < nT() ? nD() : nDT(), i < nT() ? nZ() : nX());
		}

		VectorMap dMin(unsigned i)
		{
			assert(i <= nT());
			return VectorMap(_dMin.data() + i * nD(), i < nT() ? nD() : nDT());
		}

		VectorConstMap dMin(unsigned i) const
		{
			assert(i <= nT());
			return VectorConstMap(_dMin.data() + i * nD(), i < nT() ? nD() : nDT());
		}

		VectorMap dMax(unsigned i)
		{
			assert(i <= nT());
			return VectorMap(_dMax.data() + i * nD(), i < nT() ? nD() : nDT());
		}

		VectorConstMap dMax(unsigned i) const
		{
			assert(i <= nT());
			return VectorConstMap(_dMax.data() + i * nD(), i < nT() ? nD() : nDT());
		}
				
		VectorMap c(unsigned i)
		{
			return VectorMap(_c.data() + i * nX(), nX());
		}

		VectorConstMap c(unsigned i) const
		{
			return VectorConstMap(_c.data() + i * nX(), nX());
		}
		
		VectorMap zMin(unsigned i)
		{
			assert(i < nT() + 1);
			return VectorMap(_zMin.data() + i * nZ(), i < nT() ? nZ() : nX());
		}

		VectorConstMap zMin(unsigned i) const
		{
			assert(i < nT() + 1);
			return VectorConstMap(_zMin.data() + i * nZ(), i < nT() ? nZ() : nX());
		}
		
		VectorMap zMax(unsigned i)
		{
			assert(i < nT() + 1);
			return VectorMap(_zMax.data() + i * nZ(), i < nT() ? nZ() : nX());
		}

		VectorConstMap zMax(unsigned i) const
		{
			assert(i < nT() + 1);
			return VectorConstMap(_zMax.data() + i * nZ(), i < nT() ? nZ() : nX());
		}

		VectorMap xMin(unsigned i)
		{
			assert(i < nT() + 1);
			return VectorMap(_zMin.data() + i * nZ(), nX());
		}

		VectorConstMap xMin(unsigned i) const
		{
			assert(i < nT() + 1);
			return VectorConstMap(_zMin.data() + i * nZ(), nX());
		}

		VectorMap xMax(unsigned i)
		{
			assert(i < nT() + 1);
			return VectorMap(_zMax.data() + i * nZ(), nX());
		}

		VectorConstMap xMax(unsigned i) const
		{
			assert(i < nT() + 1);
			return VectorConstMap(_zMax.data() + i * nZ(), nX());
		}

		VectorMap uMin(unsigned i)
		{
			assert(i < nT());
			return VectorMap(_zMin.data() + i * nZ() + nX(), nU());
		}

		VectorConstMap uMin(unsigned i) const
		{
			assert(i < nT());
			return VectorConstMap(_zMin.data() + i * nZ() + nX(), nU());
		}

		VectorMap uMax(unsigned i)
		{
			assert(i < nT());
			return VectorMap(_zMax.data() + i * nZ() + nX(), nU());
		}

		VectorConstMap uMax(unsigned i) const
		{
			assert(i < nT());
			return VectorConstMap(_zMax.data() + i * nZ() + nX(), nU());
		}

	private:
		MultiStageQP(const MultiStageQPSize& size)
		:	_size(size)
		{
			// Allocate arrays.
			_H.resize(nZ() * nZ() * nT() + nX() * nX());
			_g.resize(nZ() * nT() + nX());
			_C.resize(nX() * nZ() * nT());
			_c.resize(nX() * nT());
			_D.resize(nD() * nZ() * nT() + nDT() * nX());
			_dMin.resize(nD() * nT() + nDT());
			_dMax.resize(nD() * nT() + nDT());
			_zMin.resize(nZ() * nT() + nX());
			_zMax.resize(nZ() * nT() + nX());
			_zOpt.resize(nZ() * nT() + nX());
		}

		// Private data members.
		//

		const MultiStageQPSize _size;

		// _H stores _Nt row-major matrices of size _Nz x _Nz and 1 matrix of size _Nx x _Nx.
		std::vector<double> _H;

		// _g stores _Nt vectors of size _Nz and 1 vector of size _Nx
		std::vector<double> _g;

		// _C stores _Nt row-major matrices of size _Nx x _Nz
		std::vector<double> _C;

		// _c stores _Nt vectors of size _Nx
		std::vector<double> _c;

		// _D stores _Nt row-major matrices of size _Nd x _Nz and 1 row-major matrix of size _NdT x _Nx
		std::vector<double> _D;

		// _dMin stores _Nt vectors of size _Nd and 1 vector of size _NdT
		std::vector<double> _dMin;

		// _dMax stores _Nt vectors of size _Nd and 1 vector of size _NdT
		std::vector<double> _dMax;

		// _zMin stores _Nt vectors of size _Nz and 1 vector of size _Nx
		std::vector<double> _zMin;

		// _zMax stores _Nt vectors of size _Nz and 1 vector of size _Nx
		std::vector<double> _zMax;

		// Primal optimal solution.
		// _zOpt stores _Nt vectors of size _Nz and 1 vector of size _Nx
		std::vector<double> _zOpt;
	};

	inline void MultiStageQP::PrintQP_C(std::ostream& log_stream) const
	{
		using std::endl;

		Eigen::IOFormat C_format(Eigen::StreamPrecision, 0, ", ", ",\n", "", "", "", "");

		log_stream << "const double H[] = {" << endl;
		for (unsigned i = 0; i <= nT(); ++i)
			log_stream << H(i).format(C_format) << "," << endl;
		log_stream << "};" << endl << endl;

		log_stream << "const double g[] = {" << endl;
		for (unsigned i = 0; i <= nT(); ++i)
			log_stream << g(i).transpose().format(C_format) << "," << endl;
		log_stream << "};" << endl << endl;

		log_stream << "const double C[] = {" << endl;
		for (unsigned i = 0; i < nT(); ++i)
			log_stream << C(i).format(C_format) << "," << endl;
		log_stream << "};" << endl << endl;

		log_stream << "const double c[] = {" << endl;
		for (unsigned i = 0; i < nT(); ++i)
			log_stream << c(i).transpose().format(C_format) << "," << endl;
		log_stream << "};" << endl << endl;

		log_stream << "const double D[] = {" << endl;
		for (unsigned i = 0; i <= nT(); ++i)
			log_stream << D(i).format(C_format) << "," << endl;
		log_stream << "};" << endl << endl;

		log_stream << "const double dMin[] = {" << endl;
		for (unsigned i = 0; i <= nT(); ++i)
			log_stream << dMin(i).transpose().format(C_format) << ",";
		log_stream << "};" << endl << endl;

		log_stream << "const double dMax[] = {" << endl;
		for (unsigned i = 0; i <= nT(); ++i)
			log_stream << dMax(i).transpose().format(C_format) << ",";
		log_stream << "};" << endl << endl;

		PrintQP_zMin_C(log_stream);
		PrintQP_zMax_C(log_stream);
	}

	inline void MultiStageQP::PrintQP_MATLAB(std::ostream& log_stream, const std::string& var_name) const
	{
		using std::endl;

		for (unsigned k = 0; k <= nT(); ++k)
		{
			log_stream << var_name << ".H{" << k + 1 << "} = [..." << endl << H(k) << "];" << endl;
			log_stream << var_name << ".g{" << k + 1 << "} = [..." << endl << g(k) << "];" << endl;

			if (k < nT())
			{
				log_stream << var_name << ".C{" << k + 1 << "} = [..." << endl << C(k) << "];" << endl;
				log_stream << var_name << ".c{" << k + 1 << "} = [..." << endl << c(k) << "];" << endl;
			}

			log_stream << var_name << ".D{" << k + 1 << "} = [..." << endl << D(k) << "];" << endl;
			log_stream << var_name << ".dMin{" << k + 1 << "} = [..." << endl << dMin(k) << "];" << endl;
			log_stream << var_name << ".dMax{" << k + 1 << "} = [..." << endl << dMax(k) << "];" << endl;

			log_stream << var_name << ".zMin{" << k + 1 << "} = [..." << endl << zMin(k) << "];" << endl;
			log_stream << var_name << ".zMax{" << k + 1 << "} = [..." << endl << zMax(k) << "];" << endl;
		}
	}

	inline void MultiStageQP::PrintQP_zMin_C(std::ostream& log_stream) const
	{
		using std::endl;

		Eigen::IOFormat C_format(Eigen::StreamPrecision, 0, ", ", ",\n", "", "", "", "");

		log_stream << "const double zLow[] = {" << endl;
		for (unsigned i = 0; i <= nT(); ++i)
			log_stream << zMin(i).transpose().format(C_format) << "," << endl;
		log_stream << endl << "};" << endl << endl;
	}

	inline void MultiStageQP::PrintQP_zMax_C(std::ostream& log_stream) const
	{
		using std::endl;

		Eigen::IOFormat C_format(Eigen::StreamPrecision, 0, ", ", ",\n", "", "", "", "");

		log_stream << "const double zUpp[] = {" << endl;
		for (unsigned i = 0; i <= nT(); ++i)
			log_stream << zMax(i).transpose().format(C_format) << "," << endl;
		log_stream << endl << "};" << endl << endl;
	}
}
