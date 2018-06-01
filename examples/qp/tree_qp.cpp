#include <boost/config.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/wavefront.hpp>

#include <tmpc/ocp/OcpSizeGraph.hpp>
#include <tmpc/core/IteratorRange.hpp>


int main()
{
    using namespace boost;
    using namespace tmpc;

    size_t const N = 4;
    OcpSizeGraph g(N);
    add_edge(0, 1, g);
    add_edge(1, 2, g);
    add_edge(0, 3, g);

    auto index_map = get(vertex_index, g);

    for (auto i : make_iterator_range(vertices(g)))
    {
        std::cout << get(index_map, i);
        auto adj_vert = make_iterator_range(adjacent_vertices(i, g));
        if (adj_vert.size() == 0)
            std::cout << " has no children";
        else
            std::cout << " is the parent of ";
        for (auto ai = adj_vert.begin(); ai != adj_vert.end(); ++ai) 
        {
            std::cout << get(index_map, *ai);
            if (std::next(ai) != adj_vert.end())
                std::cout << ", ";
        }
        std::cout << std::endl;
    }

    auto ocp_size = get(OcpSizeProperty_t(), g);
    put(ocp_size, 0, OcpSize {2, 3, 1, 1});
    put(ocp_size, 2, OcpSize {3, 4, 2, 3});

    for (auto v : make_iterator_range(vertices(g)))
    {
        auto sz = get(ocp_size, v);
        std::cout << "Vertex " << v << ", nx=" << sz.nx() << ", nu=" << sz.nu() << ", nc=" << sz.nc() << ", ns=" << sz.ns() << std::endl;
    }

    std::cout << "Max wavefront = " << max_wavefront(g) << std::endl;

    return EXIT_SUCCESS;
}