import networkx
import cfpq_data
from project.graph_op import build_cycles_graph
from project.graph_op import get_graph_info
from pydot import graph_from_dot_file
import os


def test_create_and_save_two_cycles_graphs():
    # Test of the build_cycles_graph(num_vertices1, num_vertices2, labels, filename)
    labels = ("first", "second")
    filename = "testGraph.dot"

    build_cycles_graph(52, 228, labels, filename)
    graph = graph_from_dot_file(filename)[0]
    os.remove(filename)

    g = networkx.nx_pydot.from_pydot(graph)
    cycle = networkx.find_cycle(g, orientation="original")

    assert len(g.nodes) == 281  # 52 + 228 + 1
    assert set(labels) == set(cfpq_data.get_sorted_labels(g))
    assert len(cycle) == 53


def test_get_graph_data():
    # Test of the get_graph_info(name)
    # Examples taken from https://formallanguageconstrainedpathquerying.github.io/CFPQ_Data/graphs/index.html
    (vertices, num, labels) = get_graph_info("atom")
    expected_labels = {
        "type",
        "label",
        "subClassOf",
        "comment",
        "domain",
        "range",
        "subPropertyOf",
        "creator",
        "date",
        "description",
        "format",
        "imports",
        "language",
        "publisher",
        "seeAlso",
        "title",
        "versionInfo",
    }

    assert vertices == 291
    assert num == 425
    for i in labels:
        assert i in expected_labels
