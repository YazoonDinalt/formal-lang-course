from project.task3 import AdjacencyMatrixFA
from project.automata import NondeterministicFiniteAutomaton

fin_aut = NondeterministicFiniteAutomaton()
fin_aut.add_transitions([(0, "a", 1), (0, "b", 2), (1, "c", 2), (1, "a", 0)])
fin_aut.add_start_state(0)
fin_aut.add_final_state(2)

adj = AdjacencyMatrixFA(fin_aut)


def test_accept():
    assert adj.accepts("aab")
    assert adj.accepts("aaac")
    assert adj.accepts("b")
    assert adj.accepts("aac")
    assert adj.accepts("aaab")
    assert not adj.accepts("c")


def test_empty():
    fin_aut1 = NondeterministicFiniteAutomaton()
    fin_aut1.add_transitions([(0, "a", 1), (0, "b", 2), (1, "c", 2), (1, "a", 0)])
    fin_aut1.add_start_state(0)
    fin_aut1.add_final_state(2)
    fin_aut2 = NondeterministicFiniteAutomaton()
    adj1 = AdjacencyMatrixFA(fin_aut1)
    adj2 = AdjacencyMatrixFA(fin_aut2)
    assert not adj1.is_empty()
    assert adj2.is_empty()
