import operator
from typing import Annotated

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, waiting_edge_release


class State(TypedDict):
    ran: Annotated[list, operator.add]


def _mark(name: str):
    return lambda state: {"ran": [name]}


def _count(ran: list, name: str) -> int:
    return sum(1 for entry in ran if entry == name)


def test_inclusive_edge_runs_target_once_with_arrived_subset() -> None:
    """The default edge drops the writes; the inclusive edge runs the target."""

    def build(inclusive: bool):
        g = StateGraph(State)
        for n in ["a", "b", "c", "d"]:
            g.add_node(n, _mark(n))
        g.add_conditional_edges(START, lambda s: "a", ["a", "b"])
        g.add_edge(["a", "b"], "c", inclusive=inclusive)
        g.add_edge("c", "d")
        g.add_edge("d", END)
        return g.compile()

    kept = build(False).invoke({"ran": []})
    released = build(True).invoke({"ran": []})
    assert kept["ran"] == ["a"]
    assert released["ran"] == ["a", "c", "d"]


def test_three_listed_one_selected() -> None:
    g = StateGraph(State)
    for n in ["p1", "p2", "p3", "gather"]:
        g.add_node(n, _mark(n))
    g.add_conditional_edges(START, lambda s: "p2", ["p1", "p2", "p3"])
    g.add_edge(["p1", "p2", "p3"], "gather", inclusive=True)
    g.add_edge("gather", END)
    out = g.compile().invoke({"ran": []})
    assert out["ran"] == ["p2", "gather"]


def test_no_early_fire_while_a_listed_node_is_on_its_way() -> None:
    g = StateGraph(State)
    for n in ["p1", "mid", "p2", "gather"]:
        g.add_node(n, _mark(n))
    g.add_edge(START, "p1")
    g.add_edge(START, "mid")
    g.add_edge("mid", "p2")
    g.add_edge(["p1", "p2"], "gather", inclusive=True)
    g.add_edge("gather", END)
    out = g.compile().invoke({"ran": []})
    assert _count(out["ran"], "gather") == 1
    assert out["ran"].index("p2") < out["ran"].index("gather")


def test_send_to_listed_node_holds_the_release() -> None:
    g = StateGraph(State)
    for n in ["p1", "late", "p2", "gather"]:
        g.add_node(n, _mark(n))
    g.add_edge(START, "p1")
    g.add_edge(START, "late")
    g.add_conditional_edges("late", lambda s: [Send("p2", {"ran": []})], ["p2"])
    g.add_edge(["p1", "p2"], "gather", inclusive=True)
    g.add_edge("gather", END)
    out = g.compile().invoke({"ran": []})
    assert _count(out["ran"], "gather") == 1
    assert out["ran"].index("p2") < out["ran"].index("gather")


def test_uneven_depths_once_at_every_selection_mix() -> None:
    def build():
        g = StateGraph(State)

        class P(TypedDict):
            ran: Annotated[list, operator.add]
            picked: list

        g = StateGraph(P)
        for n in ["ocr", "transcribe", "diarize", "translate", "index"]:
            g.add_node(n, _mark(n))
        g.add_conditional_edges(
            START, lambda s: s["picked"], ["ocr", "transcribe", "translate"]
        )
        g.add_edge("transcribe", "diarize")
        g.add_edge(["ocr", "diarize", "translate"], "index", inclusive=True)
        g.add_edge("index", END)
        return g.compile()

    for picked in (
        ["ocr", "transcribe", "translate"],
        ["ocr"],
        ["ocr", "transcribe"],
    ):
        out = build().invoke({"ran": [], "picked": picked})
        assert _count(out["ran"], "index") == 1, picked


def test_edge_nobody_wrote_to_stays_silent() -> None:
    g = StateGraph(State)
    for n in ["x", "a", "b", "c"]:
        g.add_node(n, _mark(n))
    g.add_conditional_edges(START, lambda s: "x", ["x", "a", "b"])
    g.add_edge(["a", "b"], "c", inclusive=True)
    g.add_edge("c", END)
    g.add_edge("x", END)
    out = g.compile().invoke({"ran": []})
    assert out["ran"] == ["x"]


def test_rearms_in_a_loop_final_incomplete_pass_still_runs_once() -> None:
    g = StateGraph(State)
    for n in ["fan", "a", "b", "merge"]:
        g.add_node(n, _mark(n))
    g.add_edge(START, "fan")
    g.add_conditional_edges(
        "fan",
        lambda s: ["a", "b"] if _count(s["ran"], "merge") < 1 else ["a"],
        ["a", "b"],
    )
    g.add_edge(["a", "b"], "merge", inclusive=True)
    g.add_conditional_edges(
        "merge",
        lambda s: "fan" if _count(s["ran"], "merge") < 2 else END,
        ["fan", END],
    )
    out = g.compile().invoke({"ran": []})
    assert _count(out["ran"], "merge") == 2


def test_cascade_resolves_in_dependency_order() -> None:
    g = StateGraph(State)
    for n in ["a", "b", "c", "j1", "j2"]:
        g.add_node(n, _mark(n))
    g.add_conditional_edges(START, lambda s: "a", ["a", "b", "c"])
    g.add_edge(["a", "b"], "j1", inclusive=True)
    g.add_edge(["j1", "c"], "j2", inclusive=True)
    g.add_edge("j2", END)
    out = g.compile().invoke({"ran": []})
    assert out["ran"] == ["a", "j1", "j2"]


def test_interrupt_at_release_point_keeps_next_honest() -> None:
    g = StateGraph(State)
    for n in ["w0", "w1", "w2", "gather"]:
        g.add_node(n, _mark(n))
    g.add_conditional_edges(START, lambda s: ["w1", "w2"], ["w0", "w1", "w2"])
    g.add_edge(["w1", "w0", "w2"], "gather", inclusive=True)
    g.add_edge("gather", END)
    app = g.compile(checkpointer=InMemorySaver(), interrupt_after=["w1"])
    config = {"configurable": {"thread_id": "parked"}}

    app.invoke({"ran": []}, config)
    paused = app.get_state(config)
    assert paused.next == ("gather",)

    result = app.invoke(None, config)
    assert _count(result["ran"], "gather") == 1
    done = app.get_state(config)
    assert done.next == ()


def test_release_record_names_arrived_and_missing() -> None:
    observed = []

    def gather(state):
        observed.append(waiting_edge_release())
        return {"ran": ["gather"]}

    def build(selection):
        g = StateGraph(State)
        g.add_node("w0", _mark("w0"))
        g.add_node("w1", _mark("w1"))
        g.add_node("gather", gather)
        g.add_conditional_edges(START, lambda s: selection, ["w0", "w1"])
        g.add_edge(["w0", "w1"], "gather", inclusive=True)
        g.add_edge("gather", END)
        return g.compile()

    build(["w1"]).invoke({"ran": []})
    build(["w0", "w1"]).invoke({"ran": []})
    assert observed[0] is not None
    assert observed[0].target == "gather"
    assert observed[0].arrived == {"w1"}
    assert observed[0].missing == {"w0"}
    assert observed[1] is None


def test_toggling_the_option_on_an_existing_thread_is_safe() -> None:
    saver = InMemorySaver()
    boom = {"on": True}

    def holder(state):
        if boom["on"]:
            boom["on"] = False
            raise RuntimeError("boom")
        return {"ran": ["holder"]}

    def build(inclusive: bool):
        g = StateGraph(State)
        g.add_node("w0", _mark("w0"))
        g.add_node("w1", _mark("w1"))
        g.add_node("mid", _mark("mid"))
        g.add_node("holder", holder)
        g.add_node("gather", _mark("gather"))
        g.add_conditional_edges(START, lambda s: ["w1", "mid"], ["w0", "w1", "mid"])
        g.add_edge("mid", "holder")
        g.add_edge(["w0", "w1"], "gather", inclusive=inclusive)
        g.add_edge("gather", END)
        g.add_edge("holder", END)
        return g.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "toggle"}}
    with pytest.raises(RuntimeError, match="boom"):
        build(False).invoke({"ran": []}, config)

    upgraded = build(True)
    result = upgraded.invoke(None, config)
    assert _count(result["ran"], "gather") == 1


def test_inclusive_with_defer_raises_at_compile() -> None:
    g = StateGraph(State)
    g.add_node("a", _mark("a"))
    g.add_node("b", _mark("b"))
    g.add_node("c", _mark("c"), defer=True)
    g.add_conditional_edges(START, lambda s: "a", ["a", "b"])
    g.add_edge(["a", "b"], "c", inclusive=True)
    g.add_edge("c", END)
    with pytest.raises(ValueError, match="defer"):
        g.compile()


def test_inclusive_on_single_start_raises() -> None:
    g = StateGraph(State)
    g.add_node("a", _mark("a"))
    g.add_node("b", _mark("b"))
    with pytest.raises(ValueError, match="list of start nodes"):
        g.add_edge("a", "b", inclusive=True)


def test_fork_from_the_pre_release_checkpoint_releases_once() -> None:
    g = StateGraph(State)
    for n in ["w0", "w1", "w2", "gather"]:
        g.add_node(n, _mark(n))
    g.add_conditional_edges(START, lambda s: ["w1", "w2"], ["w0", "w1", "w2"])
    g.add_edge(["w1", "w0", "w2"], "gather", inclusive=True)
    g.add_edge("gather", END)
    app = g.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "fork"}}

    app.invoke({"ran": []}, config)
    armed_id = None
    for snapshot in app.get_state_history(config):
        if snapshot.next == ("gather",):
            armed_id = snapshot.config["configurable"]["checkpoint_id"]
            break
    assert armed_id is not None

    fork = app.invoke(
        None, {"configurable": {"thread_id": "fork", "checkpoint_id": armed_id}}
    )
    assert _count(fork["ran"], "gather") == 1


def test_update_state_as_the_missing_node_releases_once() -> None:
    g = StateGraph(State)
    for n in ["w0", "w1", "w2", "gather"]:
        g.add_node(n, _mark(n))
    g.add_conditional_edges(START, lambda s: ["w1", "w2"], ["w0", "w1", "w2"])
    g.add_edge(["w1", "w0", "w2"], "gather", inclusive=True)
    g.add_edge("gather", END)
    app = g.compile(checkpointer=InMemorySaver(), interrupt_after=["w1"])
    config = {"configurable": {"thread_id": "update-missing"}}

    app.invoke({"ran": []}, config)
    app.update_state(config, {"ran": ["w0-manual"]}, as_node="w0")
    result = app.invoke(None, config)
    assert _count(result["ran"], "gather") == 1


@pytest.mark.anyio
async def test_ainvoke_releases_the_inclusive_edge() -> None:
    g = StateGraph(State)
    for n in ["a", "b", "c"]:
        g.add_node(n, _mark(n))
    g.add_conditional_edges(START, lambda s: "a", ["a", "b"])
    g.add_edge(["a", "b"], "c", inclusive=True)
    g.add_edge("c", END)
    out = await g.compile().ainvoke({"ran": []})
    assert out["ran"] == ["a", "c"]
