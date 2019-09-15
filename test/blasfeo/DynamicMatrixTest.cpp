#include <tmpc/blasfeo/DynamicMatrix.hpp>
#include <tmpc/Testing.hpp>


namespace tmpc :: testing
{
	TEST(DynamicMatrixTest, testDefaultCtor)
	{
		blasfeo::DynamicMatrix<double> m;

		EXPECT_EQ(rows(m), 0);
		EXPECT_EQ(columns(m), 0);
	}


	TEST(DynamicMatrixTest, testSizeCtor)
	{
		blasfeo::DynamicMatrix<double> m(2, 3);

		EXPECT_EQ(rows(m), 2);
		EXPECT_EQ(columns(m), 3);
	}


	TEST(DynamicMatrixTest, testCtorFromBlazeMatrix)
	{
		blaze::DynamicMatrix<double, blaze::columnMajor> rhs(2, 3);
		randomize(rhs);

		blasfeo::DynamicMatrix<double> lhs(rhs);
		EXPECT_EQ(rows(lhs), rows(rhs));
		EXPECT_EQ(columns(lhs), columns(rhs));

		for (size_t i = 0; i < rows(lhs); ++i)
			for (size_t j = 0; j < columns(lhs); ++j)
				EXPECT_EQ(lhs(i, j), rhs(i, j)) << "element mismatch at index (" << i << ", " << j << ")";
	}


	TEST(DynamicMatrixTest, testRows)
	{
		blasfeo::DynamicMatrix<double> m(2, 3);
		EXPECT_EQ(rows(m), 2);
	}


	TEST(DynamicMatrixTest, testColumns)
	{
		blasfeo::DynamicMatrix<double> m(2, 3);
		EXPECT_EQ(columns(m), 3);
	}


	TEST(DynamicMatrixTest, testResize)
	{
		blasfeo::DynamicMatrix<double> m(2, 3);
		ASSERT_EQ(rows(m), 2);
		ASSERT_EQ(columns(m), 3);

		m.resize(4, 5);
		ASSERT_EQ(rows(m), 4);
		ASSERT_EQ(columns(m), 5);

		// Assign to the last element to check that there is enough memory s.t. we don't SEGFAULT
		m(3, 4) = 42.;
	}


	TEST(DynamicMatrixTest, testElementAccess)
	{
		blasfeo::DynamicMatrix<double> m(2, 3);
		ASSERT_EQ(rows(m), 2);
		ASSERT_EQ(columns(m), 3);

		for (size_t i = 0; i < rows(m); ++i)
			for (size_t j = 0; j < columns(m); ++j)
				m(i, j) = 10 * i + j;

		for (size_t i = 0; i < rows(m); ++i)
			for (size_t j = 0; j < columns(m); ++j)
				EXPECT_EQ(m(i, j), 10 * i + j) << "element mismatch at index (" << i << ", " << j << ")";
	}


	TEST(DynamicMatrixTest, testAssignBlazeMatrix)
	{
		blasfeo::DynamicMatrix<double> lhs;
		blaze::DynamicMatrix<double, blaze::columnMajor> rhs(2, 3);
		randomize(rhs);

		lhs = std::as_const(rhs);
		EXPECT_EQ(rows(lhs), rows(rhs));
		EXPECT_EQ(columns(lhs), columns(rhs));

		for (size_t i = 0; i < rows(lhs); ++i)
			for (size_t j = 0; j < columns(lhs); ++j)
				EXPECT_EQ(lhs(i, j), rhs(i, j)) << "element mismatch at index (" << i << ", " << j << ")";
	}


	TEST(DynamicMatrixTest, testUnpack)
	{
		blasfeo::DynamicMatrix<double> B;
		blaze::DynamicMatrix<double, blaze::columnMajor> A0(2, 3), A1;
		randomize(A0);

		B = std::as_const(A0);
		B.unpack(A1);

		TMPC_EXPECT_EQ(A1, A0);
	}
}