import cfpq_data
from networkx.drawing import nx_pydot


def get_graph_info(name):
    gr_path = cfpq_data.download(name)
    graph = cfpq_data.graph_from_csv(gr_path)

    return (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        cfpq_data.get_sorted_labels(graph),
    )


def build_cycles_graph(num_vertices1, num_vertices2, labels, filename):
    g = cfpq_data.labeled_two_cycles_graph(num_vertices1, num_vertices2, labels=labels)

    graph_to_pydot = nx_pydot.to_pydot(g)
    graph_to_pydot.write(filename)
