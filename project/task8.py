import networkx as nx
from pyformlang.cfg import CFG
from pyformlang.rsa import RecursiveAutomaton
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State
from typing import Iterable
from scipy.sparse import csc_matrix

from project.task3 import AdjacencyMatrixFA, intersect_automata
from project.automata import graph_to_nfa


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)


def rsm_to_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    for nonterm, box in rsm.boxes.items():
        dfa = box.dfa

        for st in dfa.start_states:
            nfa.add_start_state(State((nonterm, st.value)))
        for st in dfa.final_states:
            nfa.add_final_state(State((nonterm, st.value)))

        for from_st in dfa.states:
            transitions = dfa.to_dict().get(from_st)
            if transitions is None:
                continue
            for symbol, to_st in transitions.items():
                # Обработка множественных переходов
                if isinstance(to_st, Iterable):
                    for st in to_st:
                        nfa.add_transition(
                            State((nonterm, from_st)), symbol, State((nonterm, st))
                        )
                else:
                    nfa.add_transition(
                        State((nonterm, from_st)), symbol, State((nonterm, to_st))
                    )

    return nfa


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    rsm_matrix = AdjacencyMatrixFA(rsm_to_nfa(rsm))
    graph_matrix = AdjacencyMatrixFA(
        graph_to_nfa(nx.MultiDiGraph(graph), start_nodes, final_nodes)
    )

    while True:
        transitive_closure = intersect_automata(
            graph_matrix, rsm_matrix
        ).transitive_closure()
        delta = {}

        for i, j in zip(*transitive_closure.nonzero()):
            rsm_i, rsm_j = i % rsm_matrix.states_count, j % rsm_matrix.states_count
            st1, st2 = rsm_matrix.id_state[rsm_i], rsm_matrix.id_state[rsm_j]

            if st1 in rsm_matrix.start_states and st2 in rsm_matrix.final_states:
                assert st1.value[0] == st2.value[0]
                nonterm = st1.value[0]
                graph_i, graph_j = (
                    i // rsm_matrix.states_count,
                    j // rsm_matrix.states_count,
                )

                if (
                    nonterm in graph_matrix.decomposition
                    and graph_matrix.decomposition[nonterm][graph_i, graph_j]
                ):
                    continue

                if nonterm not in delta:
                    delta[nonterm] = csc_matrix(
                        (graph_matrix.states_count, graph_matrix.states_count),
                        dtype=bool,
                    )
                delta[nonterm][graph_i, graph_j] = True

        if not delta:
            break

        for symbol in delta.keys():
            if symbol not in graph_matrix.decomposition:
                graph_matrix.decomposition[symbol] = delta[symbol]
            else:
                graph_matrix.decomposition[symbol] += delta[symbol]

    start_matrix = graph_matrix.decomposition.get(rsm.initial_label)

    if start_matrix is None:
        return set()

    result = set()
    for start in start_nodes:
        for final in final_nodes:
            if start_matrix[
                graph_matrix.state_id[State(start)], graph_matrix.state_id[State(final)]
            ]:
                result.add((start, final))

    return result
