from typing import Union
import networkx as nx
from graphviz import Digraph


def nx_to_graphviz(nx_graph: Union[nx.DiGraph, nx.MultiDiGraph]):
    graph = Digraph(comment=nx_graph.graph['name'])
    for ci in nx_graph.nodes():
        cnode = nx_graph.nodes[ci]
        lineno = cnode['line_num']
        source = cnode['code']
        shape, peripheries = 'oval', '1'
        if cnode['type'] == 'Condition':
            shape = 'diamond'
        if cnode['type'] in ('CFGEntryNode', 'CFGExitNode'):
            shape, peripheries = 'oval', '2'
        if cnode['type'] in ('ReturnStatement', 'ExpressionStatement', 'IdentifierDeclStatement'):
            shape = 'rectangle'
        new_node_name = "%s: line %d # %s" % (str(ci), lineno, source)
        graph.node(str(ci), label=new_node_name, shape=shape, peripheries=peripheries)
    if isinstance(nx_graph, nx.MultiDiGraph):
        colors = {'FLOWS_TO': 'blue', 'REACHES': 'red'}
        for ei in nx_graph.edges:
            cedge = nx_graph.edges[ei]
            type = cedge['e_type']
            if type == 'REACHES':
                var = cedge['var']
                graph.edge(str(ei[0]), str(ei[1]), color=colors[cedge['e_type']], label=var)
            else:
                graph.edge(str(ei[0]), str(ei[1]), color=colors[cedge['e_type']])
    else:
        for ei in nx_graph.edges:
            graph.edge(str(ei[0]), str(ei[1]))
    return graph


def save_graph_dot(nx_graph, save_path):
    graphviz_g = nx_to_graphviz(nx_graph)
    graphviz_g.save(save_path)
    return graphviz_g
