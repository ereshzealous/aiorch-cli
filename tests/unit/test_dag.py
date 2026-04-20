import pytest

from aiorch.core.dag import DAGError, build_graph, get_execution_order
from aiorch.core.parser import parse_string


def _pipeline(yaml_src: str):
    return parse_string(yaml_src)


def test_build_graph_linear_chain_has_expected_edges():
    af = _pipeline(
        """
        name: chain
        steps:
          a:
            run: echo a
          b:
            run: echo b
            depends: [a]
          c:
            run: echo c
            depends: [b]
        """
    )
    graph = build_graph(af)
    assert graph == {"a": set(), "b": {"a"}, "c": {"b"}}


def test_build_graph_rejects_unknown_dependency():
    af = _pipeline(
        """
        name: bad-dep
        steps:
          a:
            run: echo a
            depends: [ghost]
        """
    )
    with pytest.raises(DAGError):
        build_graph(af)


def test_get_execution_order_layers_independent_steps_together():
    af = _pipeline(
        """
        name: parallel
        steps:
          a:
            run: echo a
          b:
            run: echo b
          join:
            run: echo done
            depends: [a, b]
        """
    )
    layers = get_execution_order(build_graph(af))
    assert {frozenset(layer) for layer in layers} == {frozenset({"a", "b"}), frozenset({"join"})}
    assert layers[-1] == ["join"]


def test_get_execution_order_detects_cycle():
    graph = {"a": {"b"}, "b": {"a"}}
    with pytest.raises(DAGError):
        get_execution_order(graph)
