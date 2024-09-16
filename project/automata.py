from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    State,
    DeterministicFiniteAutomaton,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    dfa = Regex(regex).to_epsilon_nfa().to_deterministic()

    return dfa.minimize()


def graph_to_nfa(graph, start_states, final_states) -> NondeterministicFiniteAutomaton:
    states = set(int(state) for state in graph.nodes)
    nfa = NondeterministicFiniteAutomaton().from_networkx(graph)
    start_states = states if not start_states else start_states
    final_states = states if not final_states else final_states

    for state in start_states:
        nfa.add_start_state(State(state))

    for state in final_states:
        nfa.add_final_state(State(state))

    nfa.remove_epsilon_transitions()

    return nfa
