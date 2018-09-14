#include <tmpc/ocp/OcpGraph.hpp>
#include <tmpc/core/Range.hpp>
#include <tmpc/core/PropertyMap.hpp>

#include <tmpc/test_tools.hpp>

#include <array>


namespace tmpc :: testing
{
	// Build a small tree
    //
    //     1 - 3
    //   /
    // 0 - 2 - 4
	TEST(OcpGraphTest, test_ocpGraphFromOutDegree)
	{
		size_t const N = 5;
		std::array<size_t, N> const ns = { 2, 1, 1, 0, 0, };

		OcpGraph const g = ocpGraphFromOutDegree(ns.begin());
		auto const edge_id = get(edge_index, g);
	
		EXPECT_EQ(num_vertices(g), N);
		EXPECT_EQ(num_edges(g), N - 1);

		EXPECT_EQ(out_degree(0, g), 2);
		EXPECT_EQ(out_degree(1, g), 1);
		EXPECT_EQ(out_degree(2, g), 1);
		EXPECT_EQ(out_degree(3, g), 0);
		EXPECT_EQ(out_degree(4, g), 0);

		EXPECT_EQ(adjacent_vertices(0, g)[0], 1);
		EXPECT_EQ(adjacent_vertices(0, g)[1], 2);
		EXPECT_EQ(adjacent_vertices(1, g)[0], 3);
		EXPECT_EQ(adjacent_vertices(2, g)[0], 4);

		EXPECT_EQ(edge_id[out_edges(0, g)[0]], 0);
		EXPECT_EQ(edge_id[out_edges(0, g)[1]], 1);
		EXPECT_EQ(edge_id[out_edges(1, g)[0]], 2);
		EXPECT_EQ(edge_id[out_edges(2, g)[0]], 3);
	}


	// Build a linear graph
    //
    // 0 - 1 - 2 - 3
	TEST(OcpGraphTest, test_ocpGraphLinear)
	{
		size_t const N = 4;
		OcpGraph const g = ocpGraphLinear(N);
		auto const edge_id = get(edge_index, g);
	
		EXPECT_EQ(num_vertices(g), N);
		EXPECT_EQ(num_edges(g), N - 1);

		EXPECT_EQ(out_degree(0, g), 1);
		EXPECT_EQ(out_degree(1, g), 1);
		EXPECT_EQ(out_degree(2, g), 1);
		EXPECT_EQ(out_degree(3, g), 0);

		EXPECT_EQ(adjacent_vertices(0, g).front(), 1);
		EXPECT_EQ(adjacent_vertices(1, g).front(), 2);
		EXPECT_EQ(adjacent_vertices(2, g).front(), 3);

		EXPECT_EQ(adjacent_vertices(0, g).begin() + 1, adjacent_vertices(0, g).end());
		EXPECT_EQ(adjacent_vertices(1, g).begin() + 1, adjacent_vertices(1, g).end());
		EXPECT_EQ(adjacent_vertices(2, g).begin() + 1, adjacent_vertices(2, g).end());
		EXPECT_EQ(adjacent_vertices(3, g).begin(), adjacent_vertices(3, g).end());

		EXPECT_EQ(edge_id[out_edges(0, g).front()], 0);
		EXPECT_EQ(edge_id[out_edges(1, g).front()], 1);
		EXPECT_EQ(edge_id[out_edges(2, g).front()], 2);
	}
}