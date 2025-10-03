import networkx as nx
import numpy as np
from pyformlang.finite_automaton import Symbol
from scipy.sparse import csr_matrix

from project.automata import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA


def front(regex_fa: AdjacencyMatrixFA, graph_fa: AdjacencyMatrixFA) -> csr_matrix:
    current_front = csr_matrix(
        (
            regex_fa.states_count * len(graph_fa.start_states),
            graph_fa.states_count,
        )
    )

    graph_start_states = {
        state
        for state, _ in sorted(
            graph_fa.start_states_id.items(), key=lambda item: item[1]
        )
    }
    for i, graph_state in enumerate(graph_start_states):
        for regex_state in regex_fa.start_states:
            regex_fa_state = regex_fa.state_id[regex_state] + i * regex_fa.states_count
            graph_fa_state = graph_fa.state_id[graph_state]
            current_front[regex_fa_state, graph_fa_state] = True

    return current_front


def create_next_front(
    current_front: csr_matrix,
    regex_fa: AdjacencyMatrixFA,
    graph_fa: AdjacencyMatrixFA,
    symbols: set[Symbol],
) -> csr_matrix:
    number_of_states = regex_fa.states_count
    for s in symbols:
        next_front = current_front @ graph_fa.decomposition[s]
        for i in range(len(graph_fa.start_states)):
            indices = np.arange(i * number_of_states, (i + 1) * number_of_states)
            next_front[indices] = regex_fa.decomposition[s].T @ next_front[indices]
        current_front += next_front
    return current_front


def ms_bfs_based_rpq(
    regex: str,
    graph: nx.MultiDiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    regex_dfa = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    symbols = set(regex_dfa.decomposition.keys()).intersection(
        graph_nfa.decomposition.keys()
    )
    current_front = front(regex_dfa, graph_nfa)
    visited = current_front
    while current_front.size > 0:
        next_front = create_next_front(current_front, regex_dfa, graph_nfa, symbols)
        current_front = next_front > visited
        visited += current_front

    pairs = set()
    graph_start_states = {
        state
        for state, _ in sorted(
            graph_nfa.start_states_id.items(), key=lambda item: item[1]
        )
    }
    graph_dict = {v: k for k, v in graph_nfa.state_id.items()}

    for i, graph_start in enumerate(graph_start_states):
        start = i * regex_dfa.states_count
        end = (i + 1) * regex_dfa.states_count
        for regex_final in regex_dfa.final_states:
            for visit_i in (
                visited[start:end].getrow(regex_dfa.state_id[regex_final]).indices
            ):
                if graph_dict[visit_i] in graph_nfa.final_states:
                    pairs.add((graph_start, graph_dict[visit_i]))

    return pairs
