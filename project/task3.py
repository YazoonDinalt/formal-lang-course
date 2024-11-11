from project.automata import NondeterministicFiniteAutomaton, regex_to_dfa, graph_to_nfa
from collections.abc import Iterable
from pyformlang.finite_automaton import Symbol, State
from networkx import MultiDiGraph
from scipy.sparse import csc_matrix, kron
from itertools import product
import numpy as np


class AdjacencyMatrixFA:
    def __init__(self, automation: NondeterministicFiniteAutomaton = None):
        self.matrix = {}

        if automation is None:
            self.states_count = 0
            self.state_id = dict()
            self.id_state = dict()
            self.start_states = set()
            self.start_states_id = dict()
            self.final_states = set()
            self.decomposition = dict()
            return

        self.states_count = len(automation.states)
        self.state_id: dict[State, int] = {
            st: i for i, st in enumerate(automation.states)
        }
        self.id_state: dict[int, State] = {i: st for st, i in self.state_id.items()}
        self.start_states: set[State] = automation.start_states
        self.start_states_id: dict[State, int] = {
            st: i for i, st in enumerate(automation.start_states)
        }
        self.final_states: set[State] = automation.final_states
        self.decomposition: dict[Symbol, csc_matrix] = {}

        for state in self.state_id.keys():
            id = self.state_id[state]
            transitions: dict[Symbol, State | set[State]] = automation.to_dict().get(
                state
            )
            if transitions is None:
                continue
            for symbol in transitions.keys():
                if symbol not in self.decomposition:
                    self.decomposition[symbol] = csc_matrix(
                        (self.states_count, self.states_count), dtype=bool
                    )
                if isinstance(transitions[symbol], Iterable):
                    for to_st in transitions[symbol]:
                        to_idx = self.state_id[to_st]
                        self.decomposition[symbol][id, to_idx] = True
                else:
                    to_st: State = transitions[symbol]
                    to_idx = self.state_id[to_st]
                    self.decomposition[symbol][id, to_idx] = True

    def accepts(self, word: Iterable[Symbol]) -> bool:
        states = set(self.start_states)

        for letter in word:
            if self.decomposition.get(letter) is None:
                return False

            for s1, s2 in product(states, self.state_id.keys()):
                if self.decomposition[letter][self.state_id[s1], self.state_id[s2]]:
                    states.add(s2)

        if states.intersection(self.final_states):
            return True

        return False

    def is_empty(self) -> bool:
        transitive_closure = self.transitive_closure()
        for start_state in self.start_states:
            for final_state in self.final_states:
                if transitive_closure[
                    self.state_id[start_state], self.state_id[final_state]
                ]:
                    return False

        return True

    def transitive_closure(self) -> np.ndarray:
        A = np.eye(self.states_count, dtype=bool)

        for dec in self.decomposition.values():
            A |= dec.toarray()

        transitive_closure = np.linalg.matrix_power(A, self.states_count).astype(bool)
        return transitive_closure


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    intersection_matrix = AdjacencyMatrixFA()

    intersection_matrix.states_count = automaton1.states_count * automaton2.states_count

    for s1 in automaton1.state_id.keys():
        for s2 in automaton2.state_id.keys():
            id1, id2 = automaton1.state_id[s1], automaton2.state_id[s2]
            intesection_id = id1 * automaton2.states_count + id2

            intersection_matrix.state_id[State((s1, s2))] = intesection_id
            if s1 in automaton1.start_states and s2 in automaton2.start_states:
                intersection_matrix.start_states.add(State((s1, s2)))
            if s1 in automaton1.final_states and s2 in automaton2.final_states:
                intersection_matrix.final_states.add(State((s1, s2)))

    intersection_matrix.decomposition = {
        key: kron(
            automaton1.decomposition[key],
            automaton2.decomposition[key],
            format="csr",
        )
        for key in automaton1.decomposition.keys()
        if key in automaton2.decomposition
    }

    return intersection_matrix


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_to_matrix = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_to_matrix = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    intersection = intersect_automata(regex_to_matrix, graph_to_matrix)
    closure = intersection.transitive_closure()
    return {
        (graph_start, graph_final)
        for graph_start in graph_to_matrix.start_states
        for graph_final in graph_to_matrix.final_states
        if any(
            closure[
                intersection.state_id[(regex_start, graph_start)],
                intersection.state_id[(regex_final, graph_final)],
            ]
            for regex_start in regex_to_matrix.start_states
            for regex_final in regex_to_matrix.final_states
        )
    }
