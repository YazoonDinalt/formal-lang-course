from functools import reduce
from itertools import product
import networkx as nx
from pyformlang.finite_automaton import Symbol
import numpy as np
from numpy import bool_
from scipy.sparse import csr_array, csr_matrix

from project.automata import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA
from numpy.typing import NDArray


def create_array(n: int, m: int, i: int, j: int) -> NDArray[bool_]:
    a = np.zeros((n, m), dtype=bool_)
    a[i, j] = True
    return a


def create_initial_front(n: int, m: int, start_states: set[int]) -> csr_matrix:
    arrays = [[create_array(n, m, i, j)] for (i, j) in start_states]
    return csr_matrix(np.block(arrays))


def get_from_block(M: csr_matrix, n: int, m: int, k: int, i: int, j: int) -> bool_:
    return M[n * k + i, j]


def update_front(
  front: csr_matrix,
  dfa_boolean_decomposition: dict[Symbol, csr_array],
  nfa_boolean_decomposition: dict[Symbol, csr_array],
  k: int,
  n: int,
  symbols: set[Symbol],
) -> csr_matrix:
  decomposed_fronts = {symbol: front @ nfa_boolean_decomposition[symbol] for symbol in symbols}

  for symbol, decomposed_front in decomposed_fronts.items():
    for i in range(k):
      start_idx, end_idx = n * i, n * (i + 1)
      decomposed_front[start_idx:end_idx] = dfa_boolean_decomposition[symbol].T @ decomposed_front[start_idx:end_idx]

  result = reduce(
    lambda acc, val: acc + val, decomposed_fronts.values(), csr_matrix(front.shape)
  )

  return result


def ms_bfs_based_rpq(
    regex: str,
    graph: nx.MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    dfa_from_regex = regex_to_dfa(regex)
    nfa_from_graph = graph_to_nfa(graph, start_nodes, final_nodes)
    dfa_matrix = AdjacencyMatrixFA(dfa_from_regex)
    nfa_matrix = AdjacencyMatrixFA(nfa_from_graph)
    dfa_state_count = len(dfa_matrix.states)
    nfa_state_count = len(nfa_matrix.states)
    nfa_index_to_state_map = {idx: state for state, idx in nfa_matrix.states.items()}
    start_state_pairs = product(dfa_matrix.start_states, nfa_matrix.start_states)
    pair_count = len(dfa_matrix.start_states) * len(nfa_matrix.start_states)
    common_alphabet = dfa_matrix.boolean_decomposition.keys() & nfa_matrix.boolean_decomposition.keys()

    current_front = create_initial_front(dfa_state_count, nfa_state_count, start_state_pairs)
    explored_states = current_front.copy()
    while current_front.nnz != 0:
        next_front = update_front(
            current_front,
            dfa_matrix.boolean_decomposition,
            nfa_matrix.boolean_decomposition,
            pair_count,
            dfa_state_count,
            common_alphabet,
        )
        unvisited_states = next_front > explored_states
        explored_states += unvisited_states
        current_front = unvisited_states

    final_result = set()
    for dfa_final_state in dfa_matrix.final_states:
        for index, nfa_start in enumerate(nfa_matrix.start_states):
            state_block = explored_states[dfa_state_count * index : dfa_state_count * (index + 1)]
            for reachable_state in state_block.getrow(dfa_final_state).indices:
                if reachable_state in nfa_matrix.final_states:
                    final_result.add(
                        (
                            nfa_index_to_state_map[nfa_start],
                            nfa_index_to_state_map[reachable_state],
                        )
                    )

    return final_result
