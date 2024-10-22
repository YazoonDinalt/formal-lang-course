from functools import reduce

import numpy as np
from networkx import MultiDiGraph
from pyformlang.finite_automaton import Symbol
from scipy.sparse import csr_matrix

from project.automata import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA


def start_front(dfa: AdjacencyMatrixFA, nfa: AdjacencyMatrixFA):
    dfa_start_state = list(dfa.start_states)[0]
    data = np.ones(len(nfa.start_states), dtype=bool)
    rows = [
        dfa_start_state + dfa.states_count * i for i in range(len(nfa.start_states))
    ]
    cols = list(nfa.start_states)

    return csr_matrix(
        (data, (rows, cols)),
        shape=(dfa.states_count * len(nfa.start_states), nfa.states_count),
        dtype=bool,
    )


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    graph_nfa = graph_to_nfa(
        graph=graph, start_states=start_nodes, final_states=final_nodes
    )

    regex_amfa = AdjacencyMatrixFA(regex_dfa)
    graph_amfa = AdjacencyMatrixFA(graph_nfa)

    init_front = start_front(regex_amfa, graph_amfa)
    result = set()
    reg_amfa_transpose_matrices = {
        symbol: matrix.transpose() for symbol, matrix in regex_amfa.matrix.items()
    }
    visited = init_front
    symbols = [
        sym for sym in regex_amfa.matrix.keys() if sym in graph_amfa.matrix.keys()
    ]

    while init_front.toarray().any():
        next_fronts: dict[Symbol, csr_matrix] = {}

        for symbol in symbols:
            next_fronts[symbol] = init_front @ graph_amfa.matrix[symbol]

            for i in range(len(graph_amfa.start_states)):
                dfa_states_cnt = regex_amfa.states_count
                start_ind, end_ind = i * dfa_states_cnt, (i + 1) * dfa_states_cnt
                next_fronts[symbol][start_ind:end_ind] = (
                    reg_amfa_transpose_matrices[symbol]
                    @ next_fronts[symbol][start_ind:end_ind]
                )

        init_front = reduce(lambda x, y: x + y, next_fronts.values(), init_front)
        init_front = init_front > visited
        visited += init_front

    reversed_nfa_states = {value: key for key, value in graph_amfa.states.items()}

    for dfa_fn_state in regex_amfa.final_states:
        for i, nfa_start_state in enumerate(graph_amfa.start_states):
            for nfa_reached in visited.getrow(
                regex_amfa.states_count * i + dfa_fn_state
            ).indices:
                if nfa_reached in graph_amfa.final_states:
                    result.add(
                        (
                            reversed_nfa_states[nfa_start_state],
                            reversed_nfa_states[nfa_reached],
                        )
                    )

    return result
