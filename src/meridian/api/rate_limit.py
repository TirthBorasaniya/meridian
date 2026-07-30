"""In-memory per-client rate limiting for the public query endpoint.

``/query`` is unauthenticated and fans out to paid Groq and Tavily APIs, so an
unbounded caller can drive cost directly. This module implements a fixed
sliding-window counter keyed by client address, held in process memory.

The limiter is deliberately simple. It is per-process, so it does not
coordinate across replicas, and it resets when the process restarts. That is
sufficient for a single-instance demo deployment; a multi-replica deployment
should place a shared limiter (for example Redis-backed) in front of the API
instead.
"""

import threading
import time
from collections import defaultdict

from fastapi import HTTPException, Request, status

from meridian.config import get_settings

# Request timestamps per client key, most recent last.
_hit_dict: dict[str, list[float]] = defaultdict(list)
_lock = threading.Lock()


def _client_key(request: Request) -> str:
    """Return the rate-limit bucket key for a request.

    Uses the peer address reported by the transport rather than a
    forwarded-for header, because that header is caller-controlled and would
    let a client trivially evade the limit by varying it. Behind a trusted
    reverse proxy every caller collapses into the proxy's address, so a proxied
    deployment should enable Uvicorn's ``--proxy-headers`` and set
    ``--forwarded-allow-ips`` so that ``request.client`` resolves correctly.

    Parameters
    ----------
    request : Request
        The incoming request.

    Returns
    -------
    str
        The client address, or ``"unknown"`` when the transport reports none.
    """
    if request.client is None:
        return "unknown"
    return request.client.host


def reset_rate_limit_state() -> None:
    """Clear all recorded request timestamps.

    Intended for tests, which would otherwise leak counter state between
    cases because the limiter is module-level.
    """
    with _lock:
        _hit_dict.clear()


def enforce_rate_limit(request: Request) -> None:
    """Reject a request that exceeds the per-client window allowance.

    Parameters
    ----------
    request : Request
        The incoming request, used to derive the client bucket key.

    Raises
    ------
    HTTPException
        With status 429 when the caller has already made
        ``rate_limit_requests`` requests inside the trailing
        ``rate_limit_window_seconds`` window.
    """
    settings = get_settings()
    window_seconds = settings.rate_limit_window_seconds
    max_requests = settings.rate_limit_requests

    if max_requests <= 0:
        return

    key = _client_key(request)
    now = time.monotonic()
    cutoff = now - window_seconds

    with _lock:
        recent_hit_list = [hit for hit in _hit_dict[key] if hit > cutoff]
        if len(recent_hit_list) >= max_requests:
            retry_after = max(1, int(recent_hit_list[0] + window_seconds - now) + 1)
            _hit_dict[key] = recent_hit_list
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: at most {max_requests} requests per "
                    f"{window_seconds} seconds."
                ),
                headers={"Retry-After": str(retry_after)},
            )
        recent_hit_list.append(now)
        _hit_dict[key] = recent_hit_list
