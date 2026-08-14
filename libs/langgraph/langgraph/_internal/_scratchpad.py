import dataclasses
from collections.abc import Callable
from typing import Any

from langgraph.types import _DC_KWARGS


@dataclasses.dataclass(**_DC_KWARGS)
class PregelScratchpad:
    step: int
    stop: int
    # call
    call_counter: Callable[[], int]
    # interrupt
    interrupt_counter: Callable[[], int]
    get_null_resume: Callable[[bool], Any]
    resume: list[Any]
    # subgraph
    subgraph_counter: Callable[[], int]
    # inclusive waiting edge release: {"target", "arrived", "missing"}
    waiting_edge_release: Any = None
