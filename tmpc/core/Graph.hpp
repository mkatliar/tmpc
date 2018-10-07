#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/copy.hpp>
#include <boost/range/iterator_range.hpp>


namespace tmpc
{
    /*
    using boost::adjacency_list;
    using boost::vecS;
    using boost::bidirectionalS;
    using boost::edge_index_t;
    using boost::property;
    using boost::no_property;
    */

    using boost::vertex_index;
    using boost::edge_index;
    using boost::default_bfs_visitor;
    using boost::default_dfs_visitor;


    template <typename Graph>
    inline auto edges(Graph const& g)
    {
        return boost::make_iterator_range(boost::edges(g));
    }


    template <typename Graph>
    inline auto vertices(Graph const& g)
    {
        return boost::make_iterator_range(boost::vertices(g));
    }


    template <typename Graph>
    inline auto in_edges(typename boost::graph_traits<Graph>::vertex_descriptor v, Graph const& g)
    {
        return boost::make_iterator_range(boost::in_edges(v, g));
    }


    template <typename Graph>
    inline auto out_edges(typename boost::graph_traits<Graph>::vertex_descriptor v, Graph const& g)
    {
        return boost::make_iterator_range(boost::out_edges(v, g));
    }


    template <typename Graph>
    inline auto adjacent_vertices(typename boost::graph_traits<Graph>::vertex_descriptor v, Graph const& g)
    {
        return boost::make_iterator_range(boost::adjacent_vertices(v, g));
    }
}