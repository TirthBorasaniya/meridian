"""Server-sent events framing and a bridge that streams a blocking agent run as async events."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

SENTINEL = object()


def sse_event(event: str, data: Any) -> str:
    """Return one SSE frame with a JSON payload."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


async def stream_blocking(
    run: Callable[[Callable[[dict], None]], Any],
) -> AsyncIterator[tuple[str, Any]]:
    """
    Run a blocking function in a worker thread and yield its events as they arrive.

    Parameters
    ----------
    run : callable
        Receives an `emit(event_dict)` callback and returns the final result.

    Yields
    ------
    item : tuple of (str, object)
        `("event", event_dict)` for each emitted event, then `("result", result)`, or
        `("error", exception)` when the function raises.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(event_dict: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ("event", event_dict))

    def target() -> None:
        try:
            result = run(emit)
            loop.call_soon_threadsafe(queue.put_nowait, ("result", result))
        except Exception as exc:  # noqa: BLE001 - forwarded to the client as an error event
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, (SENTINEL, None))

    task = loop.run_in_executor(None, target)
    while True:
        kind, value = await queue.get()
        if kind is SENTINEL:
            break
        yield kind, value
    await task
