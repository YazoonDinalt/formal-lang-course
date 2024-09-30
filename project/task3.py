from project.automata import NondeterministicFiniteAutomaton, regex_to_dfa, graph_to_nfa
import scipy.sparse as sp
from collections.abc import Iterable
from pyformlang.finite_automaton import Symbol
from networkx import MultiDiGraph


class AdjacencyMatrixFA:
    def __init__(self, automation: NondeterministicFiniteAutomaton = None):
        self.matrix = {}

        if automation is None:
            self.states = {}
            self.alphabet = set()
            self.start_states = set()
            self.final_states = set()
            return

        self.states = {st: i for i, st in enumerate(automation.states)}
        self.states_count = len(self.states)
        self.alphabet = automation.symbols

        graph = automation.to_networkx()

        self.matrix = {
            s: sp.csr_matrix((self.states_count, self.states_count), dtype=bool)
            for s in self.alphabet
        }

        for u, v, label in graph.edges(data="label"):
            if all(not s.startswith("starting_") for s in (str(u), str(v))):
                self.matrix[label][self.states[u], self.states[v]] = True

        self.start_states = {self.states[key] for key in automation.start_states}
        self.final_states = {self.states[key] for key in automation.final_states}

    def accepts(self, word: Iterable[Symbol]) -> bool:
        cf = [(list(word), st) for st in self.start_states]
        while cf:
            input_segment, state = cf.pop()
            if not input_segment and state in self.final_states:
                return True
            for next_state in self.states.values():
                if self.matrix[input_segment[0]][state, next_state]:
                    cf.append((input_segment[1:], next_state))
        return False

    def is_empty(self) -> bool:
        tc = self.transitive_closure()
        return not any(
            tc[start, final]
            for start in self.start_states
            for final in self.final_states
        )

    def transitive_closure(self):
        reach = sp.csr_matrix((self.states_count, self.states_count), dtype=bool)
        reach.setdiag(True)
        if self.matrix:
            reach = reach + sum(self.matrix.values())
        for i in range(self.states_count):
            reach = reach + (reach[:, i] * reach[i, :])

        return reach


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    intersect = AdjacencyMatrixFA()

    intersect.states_count = automaton1.states_count * automaton2.states_count

    intersect.matrix = {
        k: sp.kron(automaton1.matrix[k], automaton2.matrix[k], format="csr")
        for k in automaton1.matrix.keys()
        if k in automaton2.matrix
    }

    intersect.states = {
        (i1, i2): automaton1.states[i1] * automaton2.states_count
        + automaton2.states[i2]
        for i1 in automaton1.states.keys()
        for i2 in automaton2.states.keys()
    }

    intersect.start_states = [
        s1 * automaton2.states_count + s2
        for s1 in automaton1.start_states
        for s2 in automaton2.start_states
    ]

    intersect.final_states = [
        f1 * automaton2.states_count + f2
        for f1 in automaton1.final_states
        for f2 in automaton2.final_states
    ]

    intersect.alphabet = automaton1.alphabet.union(automaton2.alphabet)

    return intersect


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    adj_matrix_by_reg = AdjacencyMatrixFA(regex_to_dfa(regex))
    adj_matrix_by_graph = AdjacencyMatrixFA(
        graph_to_nfa(graph, start_nodes, final_nodes)
    )

    intersect = intersect_automata(adj_matrix_by_reg, adj_matrix_by_graph)

    tr_cl = intersect.transitive_closure()
    reg_raw_start_states = []
    for key, state in adj_matrix_by_reg.states.items():
        if state in adj_matrix_by_reg.start_states:
            reg_raw_start_states.append(key)

    reg_raw_final_states = []
    for key, state in adj_matrix_by_reg.states.items():
        if state in adj_matrix_by_reg.final_states:
            reg_raw_final_states.append(key)

    result = set()
    for st in start_nodes:
        for fn in final_nodes:
            for st_reg in reg_raw_start_states:
                for fn_reg in reg_raw_final_states:
                    if tr_cl[
                        intersect.states[(st_reg, st)], intersect.states[(fn_reg, fn)]
                    ]:
                        result.add((st, fn))
                        break
    return result
