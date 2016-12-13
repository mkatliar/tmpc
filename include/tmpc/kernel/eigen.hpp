#pragma once

#include <Eigen/Dense>

#include <initializer_list>

namespace tmpc
{
/**
 * \brief Class defining vector types, matrix types
 * and algebraic operations using Eigen3 library.
 *
 * \tparam Scalar_ scalar type used for matrices and vectors
 * \tparam EigenOptions options for Eigen controlling matrix layout (ColMajor/RowMajor) and alignment.
 *
 * NOTE: Can be conveniently used as a mix-in class.
 */
template <typename Scalar_, int EigenOptions = Eigen::ColMajor>
class EigenKernel
{
	template <unsigned M, unsigned N>
	struct MatrixTypeSelector
	{
		typedef Eigen::Matrix<Scalar_, M, N, EigenOptions> type;
	};

	/*
	template <unsigned M>
	struct MatrixTypeSelector<M, 1>
	{
		typedef Eigen::Matrix<Scalar_, M, 1, EigenOptions> type;
	};
	*/

	/**
	 * 1-by-N matrices must be Eigen::RowMajor, otherwise the following error is spit from Eigen:
	 * /usr/local/include/eigen3/Eigen/src/Core/PlainObjectBase.h:862:7:  error: static_assert failed "INVALID_MATRIX_TEMPLATE_PARAMETERS"
	 */
	template< unsigned N >
	struct MatrixTypeSelector<1, N>
	{
		typedef Eigen::Matrix<Scalar_, 1, N, Eigen::RowMajor> type;
	};

	/*
	template<>
	struct MatrixTypeSelector<1, 1>
	{
		typedef Eigen::Matrix<double, 1, 1, EigenOptions> type;
	};
	*/

public:
	typedef Scalar_ Scalar;
	typedef unsigned size_t;

	template <unsigned M, unsigned N>
	using Matrix = typename MatrixTypeSelector<M, N>::type;

	template <unsigned M>
	using Vector = Eigen::Matrix<Scalar, M, 1, EigenOptions>;

	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, EigenOptions> DynamicMatrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, EigenOptions> DynamicVector;

	/*
	template <typename Matrix>
	using LLT = Eigen::LLT<Matrix>;
	*/

	template <size_t M>
	static Vector<M> init_vector(std::initializer_list<Scalar> const val)
	{
		if (val.size() > M)
			throw std::invalid_argument("Invalid number of elements in Vector<> initializer list");

		Vector<M> result;
		std::copy(val.begin(), val.end(), result.data());

		return result;
	}

	static typename DynamicMatrix::ConstantReturnType constant(size_t M, size_t N, Scalar val)
	{
		return DynamicMatrix::Constant(M, N, val);
	}

	static typename DynamicVector::ConstantReturnType constant(size_t N, Scalar val)
	{
		return DynamicVector::Constant(N, val);
	}

	template <typename Matrix>
	static typename Matrix::ConstantReturnType constant(Scalar val)
	{
		return Matrix::Constant(val);
	}

	template <size_t M, size_t N>
	static typename Matrix<M, N>::ConstantReturnType constant(Scalar val)
	{
		return Matrix<M, N>::Constant(val);
	}

	template <size_t M>
	static typename Vector<M>::ConstantReturnType constant(Scalar val)
	{
		return Vector<M>::Constant(val);
	}

	static typename DynamicMatrix::ConstantReturnType zero(size_t M, size_t N)
	{
		return DynamicMatrix::Zero(M, N);
	}

	static typename DynamicVector::ConstantReturnType zero(size_t N)
	{
		return DynamicVector::Zero(N);
	}

	template <typename Matrix>
	static typename Matrix::ConstantReturnType zero()
	{
		return Matrix::Zero();
	}

	template <size_t M, size_t N>
	static typename Matrix<M, N>::ConstantReturnType zero()
	{
		return Matrix<M, N>::Zero();
	}

	template <size_t M>
	static typename Vector<M>::ConstantReturnType zero()
	{
		return Vector<M>::Zero();
	}

	template <typename Matrix>
	static void set_zero(Eigen::MatrixBase<Matrix>& m)
	{
		m.setZero();
	}

	template <typename Matrix>
	static typename Matrix::ConstantReturnType signaling_nan(typename Matrix::Index M, typename Matrix::Index N)
	{
		return Matrix::Constant(M, N, std::numeric_limits<typename Matrix::Scalar>::signaling_NaN());
	}

	template <typename Matrix>
	static typename Matrix::ConstantReturnType signaling_nan(typename Matrix::Index N)
	{
		return Matrix::Constant(N, std::numeric_limits<typename Matrix::Scalar>::signaling_NaN());
	}

	template <typename Matrix>
	static typename Matrix::ConstantReturnType signaling_nan()
	{
		return Matrix::Constant(std::numeric_limits<typename Matrix::Scalar>::signaling_NaN());
	}

	// TODO:
	// The reasons to have transpose() as a member of Kernel:
	// 1. No possibility for name conflicts
	// 2. Additional level of indirection => for example, call statistics can be gathered, etc.
	template <typename Matrix>
	static typename Matrix::ConstTransposeReturnType transpose(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.transpose();
	}

	template <typename Matrix>
	static decltype(auto) row(Eigen::MatrixBase<Matrix> const& m, typename Matrix::Index i)
	{
		return m.row(i);
	}

	template <typename Matrix>
	static decltype(auto) row(Eigen::MatrixBase<Matrix>& m, typename Matrix::Index i)
	{
		return m.row(i);
	}

	template <typename Matrix>
	static decltype(auto) col(Eigen::MatrixBase<Matrix> const& m, typename Matrix::Index i)
	{
		return m.col(i);
	}

	template <typename Matrix>
	static decltype(auto) col(Eigen::MatrixBase<Matrix>& m, typename Matrix::Index i)
	{
		return m.col(i);
	}

	template <std::size_t N, typename Matrix>
	static decltype(auto) middle_rows(Eigen::MatrixBase<Matrix> const& m, std::size_t first_row)
	{
		return m.template middleRows<N>(first_row);
	}

	template <std::size_t N, typename Matrix>
	static decltype(auto) middle_rows(Eigen::MatrixBase<Matrix>& m, std::size_t first_row)
	{
		return m.template middleRows<N>(first_row);
	}

	template <unsigned N, typename Matrix>
	static decltype(auto) top_rows(Eigen::MatrixBase<Matrix>& m)
	{
		return m.template topRows<N>();
	}

	template <unsigned N, typename Matrix>
	static decltype(auto) top_rows(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.template topRows<N>();
	}

	template <typename Matrix>
	static decltype(auto) top_rows(Eigen::MatrixBase<Matrix>& m, size_t N)
	{
		return m.template topRows(N);
	}

	template <typename Matrix>
	static decltype(auto) top_rows(Eigen::MatrixBase<Matrix> const& m, size_t N)
	{
		return m.template topRows(N);
	}

	template<unsigned N, typename Matrix>
	static decltype(auto) bottom_rows(Eigen::MatrixBase<Matrix>& m)
	{
		return m.template bottomRows<N>();
	}

	template<unsigned N, typename Matrix>
	static decltype(auto) bottom_rows(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.template bottomRows<N>();
	}

	template<unsigned N, typename Matrix>
	static decltype(auto) left_cols(Eigen::MatrixBase<Matrix>& m)
	{
		return m.template leftCols<N>();
	}

	template<unsigned N, typename Matrix>
	static decltype(auto) left_cols(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.template leftCols<N>();
	}

	template<typename Matrix>
	static decltype(auto) left_cols(Eigen::MatrixBase<Matrix>& m, size_t N)
	{
		return m.template leftCols(N);
	}

	template<typename Matrix>
	static decltype(auto) left_cols(Eigen::MatrixBase<Matrix> const& m, size_t N)
	{
		return m.template leftCols(N);
	}

	template<unsigned N, typename Matrix>
	static decltype(auto) right_cols(Eigen::MatrixBase<Matrix>& m)
	{
		return m.template rightCols<N>();
	}

	template<unsigned N, typename Matrix>
	static decltype(auto) right_cols(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.template rightCols<N>();
	}

	template<unsigned N, typename Matrix>
	static decltype(auto) middle_cols(Eigen::MatrixBase<Matrix>& m, size_t j)
	{
		return m.template middleCols<N>(j);
	}

	template<unsigned N, typename Matrix>
	static decltype(auto) middle_cols(Eigen::MatrixBase<Matrix> const& m, size_t j)
	{
		return m.template middleCols<N>(j);
	}

	template<unsigned M, unsigned N, typename Matrix>
	static decltype(auto) top_left_corner(Eigen::MatrixBase<Matrix>& m)
	{
		return m.template topLeftCorner<M, N>();
	}

	template<unsigned M, unsigned N, typename Matrix>
	static decltype(auto) top_left_corner(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.template topLeftCorner<M, N>();
	}

	template<typename Matrix>
	static decltype(auto) top_left_corner(Eigen::MatrixBase<Matrix>& m, size_t M, size_t N)
	{
		return m.template topLeftCorner(M, N);
	}

	template<typename Matrix>
	static decltype(auto) top_left_corner(Eigen::MatrixBase<Matrix> const& m, size_t M, size_t N)
	{
		return m.template topLeftCorner(M, N);
	}

	template<unsigned M, unsigned N, typename Matrix>
	static decltype(auto) top_right_corner(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.template topRightCorner<M, N>();
	}

	template <typename Matrix>
	static decltype(auto) top_right_corner(Eigen::MatrixBase<Matrix>& m, size_t M, size_t N)
	{
		return m.template topRightCorner(M, N);
	}

	template <typename Matrix>
	static decltype(auto) top_right_corner(Eigen::MatrixBase<Matrix> const& m, size_t M, size_t N)
	{
		return m.template topRightCorner(M, N);
	}

	template<unsigned M, unsigned N, typename Matrix>
	static decltype(auto) bottom_left_corner(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.template bottomLeftCorner<M, N>();
	}

	template<unsigned M, unsigned N, typename Matrix>
	static decltype(auto) bottom_right_corner(Eigen::MatrixBase<Matrix>& m)
	{
		return m.template bottomRightCorner<M, N>();
	}

	template<unsigned M, unsigned N, typename Matrix>
	static decltype(auto) bottom_right_corner(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.template bottomRightCorner<M, N>();
	}

	template<unsigned M, unsigned N, typename Matrix>
	static decltype(auto) block(Eigen::MatrixBase<Matrix>& m, size_t i, size_t j)
	{
		return m.template block<M, N>(i, j);
	}

	template<unsigned M, unsigned N, typename Matrix>
	static decltype(auto) block(Eigen::MatrixBase<Matrix> const& m, size_t i, size_t j)
	{
		return m.template block<M, N>(i, j);
	}

	template <typename Matrix>
	static std::enable_if_t<std::is_base_of<Eigen::MatrixBase<Matrix>, Matrix>::value, typename Matrix::IdentityReturnType> identity()
	{
		return Matrix::Identity();
	}

	static typename DynamicMatrix::IdentityReturnType identity(size_t M, size_t N)
	{
		return DynamicMatrix::Identity(M, N);
	}

	template <size_t M, size_t N>
	static typename Matrix<M, N>::IdentityReturnType identity()
	{
		return Matrix<M, N>::Identity();
	}

	template <typename Matrix>
	static std::enable_if_t<std::is_base_of<Eigen::MatrixBase<Matrix>, Matrix>::value, typename Matrix::Index> constexpr rows()
	{
		return Matrix::RowsAtCompileTime;
	}

	template <typename Matrix>
	static size_t rows(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.rows();
	}

	template <typename Matrix>
	static std::enable_if_t<std::is_base_of<Eigen::MatrixBase<Matrix>, Matrix>::value, typename Matrix::Index> constexpr cols()
	{
		return Matrix::ColsAtCompileTime;
	}

	template <typename Matrix>
	static size_t cols(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.cols();
	}

	template <typename Matrix>
	static typename Matrix::EvalReturnType eval(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.eval();
	}

	template <typename Matrix>
	// Eigen::NoAlias< Matrix, Eigen::MatrixBase<Matrix> > does not work as a return type
	// for the reason I which don't understand:
	// error: template argument for template template parameter must be a class template or type alias template
	// Eigen::NoAlias< Matrix, Eigen::MatrixBase<Matrix> > noalias(Eigen::MatrixBase<Matrix> const& m)
	//                         ^
	// Eigen::NoAlias< Matrix, Eigen::MatrixBase<Matrix> > noalias(Eigen::MatrixBase<Matrix> const& m)
	static decltype(auto) noalias(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.noalias();
	}

	template <typename Matrix>
	static decltype(auto) as_diagonal(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.asDiagonal();
	}

	template <typename Matrix>
	static decltype(auto) diagonal(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.diagonal();
	}

	template <typename Matrix>
	static decltype(auto) diagonal(Eigen::MatrixBase<Matrix>& m)
	{
		return m.diagonal();
	}

	template <typename Matrix>
	static decltype(auto) squared_norm(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.squaredNorm();
	}

	template <typename Matrix>
	static decltype(auto) norm_2(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.norm();
	}

	template <typename Matrix>
	static decltype(auto) inverse(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.inverse();
	}

	template <typename Matrix>
	static decltype(auto) selfadjoint_view_upper(Eigen::MatrixBase<Matrix>& m)
	{
		return m.template selfadjointView<Eigen::Upper>();
	}

	template <typename Matrix>
	static decltype(auto) selfadjoint_view_upper(Eigen::MatrixBase<Matrix>&& m)
	{
		return m.template selfadjointView<Eigen::Upper>();
	}

	template <typename Matrix>
	static decltype(auto) selfadjoint_view_upper(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.template selfadjointView<Eigen::Upper>();
	}

	template <typename Matrix>
	static decltype(auto) triangular_view_upper(Eigen::MatrixBase<Matrix>& m)
	{
		return m.template triangularView<Eigen::Upper>();
	}

	template <typename Matrix>
	static decltype(auto) triangular_view_upper(Eigen::MatrixBase<Matrix>&& m)
	{
		return m.template triangularView<Eigen::Upper>();
	}

	template <typename Matrix>
	static decltype(auto) triangular_view_upper(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.template triangularView<Eigen::Upper>();
	}

	/**
	 * \brief Returns true if at least one of the elements of m is NaN
	 */
	template <typename Matrix>
	static bool has_NaN(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.hasNaN();
	}

	/**
	 * \brief Returns true if all elements of m are finite (not NaN of inf).
	 */
	template <typename Matrix>
	static bool all_finite(Eigen::MatrixBase<Matrix> const& m)
	{
		return m.allFinite();
	}
};

}

// TODO:
// The reasons to have transpose() as a member of global namespace:
// 1. Easy syntax, no need to specify kernel. The overload of transpose() is determined by the argument type.
// 2. Kernel can be made a namespace rather than class, which will make syntax even shorter (using "using").
//
// Or should it rather be a member of Kernel, and Kernel be a namespace?
// Con: if Kernel is a namespace, it can't be parameterized.
template <typename Matrix>
static typename Matrix::ConstTransposeReturnType transpose(Eigen::MatrixBase<Matrix> const& m)
{
	return m.transpose();
}