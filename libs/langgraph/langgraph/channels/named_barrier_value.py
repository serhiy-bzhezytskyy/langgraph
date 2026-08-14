from collections.abc import Sequence
from typing import Generic

from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import EmptyChannelError, InvalidUpdateError

__all__ = (
    "InclusiveNamedBarrierValue",
    "NamedBarrierValue",
    "NamedBarrierValueAfterFinish",
)


class NamedBarrierValue(Generic[Value], BaseChannel[Value, Value, set[Value]]):
    """A channel that waits until all named values are received before making the value available."""

    __slots__ = ("names", "seen")

    names: set[Value]
    seen: set[Value]

    def __init__(self, typ: type[Value], names: set[Value]) -> None:
        super().__init__(typ)
        self.names = names
        self.seen: set[str] = set()

    def __eq__(self, value: object) -> bool:
        return isinstance(value, NamedBarrierValue) and value.names == self.names

    @property
    def ValueType(self) -> type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def copy(self) -> Self:
        """Return a copy of the channel."""
        empty = self.__class__(self.typ, self.names)
        empty.key = self.key
        empty.seen = self.seen.copy()
        return empty

    def checkpoint(self) -> set[Value]:
        return self.seen

    def from_checkpoint(self, checkpoint: set[Value]) -> Self:
        empty = self.__class__(self.typ, self.names)
        empty.key = self.key
        if checkpoint is not MISSING:
            empty.seen = checkpoint
        return empty

    def update(self, values: Sequence[Value]) -> bool:
        updated = False
        for value in values:
            if value in self.names:
                if value not in self.seen:
                    self.seen.add(value)
                    updated = True
            else:
                raise InvalidUpdateError(
                    f"At key '{self.key}': Value {value} not in {self.names}"
                )
        return updated

    def get(self) -> Value:
        if self.seen != self.names:
            raise EmptyChannelError()
        return None

    def is_available(self) -> bool:
        return self.seen == self.names

    def consume(self) -> bool:
        if self.seen == self.names:
            self.seen = set()
            return True
        return False


class NamedBarrierValueAfterFinish(
    Generic[Value], BaseChannel[Value, Value, set[Value]]
):
    """A channel that waits until all named values are received before making the value ready to be made available. It is only made available after finish() is called."""

    __slots__ = ("names", "seen", "finished")

    names: set[Value]
    seen: set[Value]

    def __init__(self, typ: type[Value], names: set[Value]) -> None:
        super().__init__(typ)
        self.names = names
        self.seen: set[str] = set()
        self.finished = False

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, NamedBarrierValueAfterFinish)
            and value.names == self.names
        )

    @property
    def ValueType(self) -> type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def copy(self) -> Self:
        """Return a copy of the channel."""
        empty = self.__class__(self.typ, self.names)
        empty.key = self.key
        empty.seen = self.seen.copy()
        empty.finished = self.finished
        return empty

    def checkpoint(self) -> tuple[set[Value], bool]:
        return (self.seen, self.finished)

    def from_checkpoint(self, checkpoint: tuple[set[Value], bool]) -> Self:
        empty = self.__class__(self.typ, self.names)
        empty.key = self.key
        if checkpoint is not MISSING:
            empty.seen, empty.finished = checkpoint
        return empty

    def update(self, values: Sequence[Value]) -> bool:
        updated = False
        for value in values:
            if value in self.names:
                if value not in self.seen:
                    self.seen.add(value)
                    updated = True
            else:
                raise InvalidUpdateError(
                    f"At key '{self.key}': Value {value} not in {self.names}"
                )
        return updated

    def get(self) -> Value:
        if not self.finished or self.seen != self.names:
            raise EmptyChannelError()
        return None

    def is_available(self) -> bool:
        return self.finished and self.seen == self.names

    def consume(self) -> bool:
        if self.finished and self.seen == self.names:
            self.finished = False
            self.seen = set()
            return True
        return False

    def finish(self) -> bool:
        if not self.finished and self.seen == self.names:
            self.finished = True
            return True
        else:
            return False


class InclusiveNamedBarrierValue(
    Generic[Value], BaseChannel[Value, Value, tuple[set[Value], bool]]
):
    """A channel that waits until all named values are received, but is released by the loop with whichever values arrived once nothing else in the run is active. Created by `add_edge` with `inclusive=True`."""

    __slots__ = ("names", "seen", "released")

    names: set[Value]
    seen: set[Value]
    released: bool

    def __init__(self, typ: type[Value], names: set[Value]) -> None:
        super().__init__(typ)
        self.names = names
        self.seen: set[str] = set()
        self.released = False

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, InclusiveNamedBarrierValue) and value.names == self.names
        )

    @property
    def ValueType(self) -> type[Value]:
        """The type of the value stored in the channel."""
        return self.typ

    @property
    def UpdateType(self) -> type[Value]:
        """The type of the update received by the channel."""
        return self.typ

    def copy(self) -> Self:
        """Return a copy of the channel."""
        empty = self.__class__(self.typ, self.names)
        empty.key = self.key
        empty.seen = self.seen.copy()
        empty.released = self.released
        return empty

    def checkpoint(self) -> tuple[set[Value], bool]:
        return (self.seen, self.released)

    def from_checkpoint(self, checkpoint: tuple[set[Value], bool] | set[Value]) -> Self:
        empty = self.__class__(self.typ, self.names)
        empty.key = self.key
        if checkpoint is not MISSING:
            # serde may round-trip the (seen, released) tuple as a list
            if isinstance(checkpoint, (tuple, list)):
                seen, released = checkpoint
                empty.seen = set(seen)
                empty.released = released
            else:
                # a bare set from a thread checkpointed before the option existed
                empty.seen = set(checkpoint)
        return empty

    def update(self, values: Sequence[Value]) -> bool:
        updated = False
        for value in values:
            if value in self.names:
                if value not in self.seen:
                    self.seen.add(value)
                    updated = True
            else:
                raise InvalidUpdateError(
                    f"At key '{self.key}': Value {value} not in {self.names}"
                )
        return updated

    def release_arrived(self) -> set[Value] | None:
        """Release with the names seen so far, or return None when there is nothing to release: an empty barrier stays silent and a complete one releases on its own."""
        if self.released or not self.seen or self.seen == self.names:
            return None
        self.released = True
        return set(self.seen)

    def get(self) -> Value:
        if not self.released and self.seen != self.names:
            raise EmptyChannelError()
        return None

    def is_available(self) -> bool:
        return self.released or self.seen == self.names

    def consume(self) -> bool:
        if self.released or self.seen == self.names:
            self.seen = set()
            self.released = False
            return True
        return False
