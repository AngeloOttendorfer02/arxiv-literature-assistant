import json
from collections.abc import Iterable, Iterator
from itertools import chain

from scilitrag.contracts import EVENT_TYPES, Model


def iter_events(lines: Iterable[str]) -> Iterator[tuple[str, Model]]:
    """Turn the lines of a server-sent event stream into validated (name, event) pairs."""
    name, data = None, []
    # A final empty line flushes an event the server did not terminate.
    for line in chain(lines, [""]):
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif line == "" and name:
            yield name, EVENT_TYPES[name].model_validate(json.loads("\n".join(data)))
            name, data = None, []
