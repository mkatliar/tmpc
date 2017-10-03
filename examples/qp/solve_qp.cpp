#include <tmpc/qp/QpOasesWorkspace.hpp>
#include <tmpc/print/qp/OcpQp.hpp>
#include <tmpc/print/ocp/OcpSolution.hpp>

#include <tmpc/BlazeKernel.hpp>
#include <tmpc/EigenKernel.hpp>
#include <tmpc/Math.hpp>

#include <vector>
#include <iostream>

int main(int, char **)
{
	using namespace tmpc;

	using Kernel = BlazeKernel<double>;
	using Workspace = QpOasesWorkspace<Kernel>;

	Workspace workspace {OcpSize {3, 0, 0}, OcpSize {0, 0, 0}};
	
	auto& stage0 = workspace.problem()[0];
	stage0.gaussNewtonCostApproximation(
		DynamicVector<Kernel> {1., 2., 42.},
		IdentityMatrix<Kernel> {3u},
		DynamicMatrix<Kernel> {3u, 0u}
	);
	stage0.bounds(-inf<double>(), -inf<double>(), inf<double>(), inf<double>());

	for (auto const& s : workspace.problem())
		std::cout << s << std::endl;

	workspace.solve();

	std::cout << workspace.solution()[0] << std::endl;

	return 0;
}