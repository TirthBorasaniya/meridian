"""Per-IP rate limiting with a per-minute and a per-day window (SPEC S5.2: 10 per minute, 100 per
day in the public demo). Fixed windows in Redis when configured, else in process memory.
"""

import threading
import time
from dataclasses import dataclass

from meridian.common.config import RateLimitConfig

MINUTE_SECONDS = 60
DAY_SECONDS = 86_400


@dataclass
class LimitDecision:
    """Whether a request may proceed, and when to retry if not."""

    o_allowed: bool
    retry_after_s: int = 0
    window: str = ""


class RateLimiter:
    """Two fixed windows per client key."""

    def __init__(self, config: RateLimitConfig, redis_client=None, clock=time.time) -> None:
        self.config = config
        self._redis = redis_client
        self._clock = clock
        self._lock = threading.Lock()
        self._count_dict: dict[str, int] = {}

    def _increment(self, key: str, ttl_seconds: int) -> int:
        if self._redis is not None:
            pipeline = self._redis.pipeline()
            pipeline.incr(key)
            pipeline.expire(key, ttl_seconds, nx=True)
            count, _ = pipeline.execute()
            return int(count)
        with self._lock:
            self._count_dict[key] = self._count_dict.get(key, 0) + 1
            return self._count_dict[key]

    def check(self, client_key: str) -> LimitDecision:
        """
        Count one request for a client and decide whether it is allowed.

        Parameters
        ----------
        client_key : str
            Client address.

        Returns
        -------
        decision : LimitDecision
            Allowed, or rejected with the seconds until the exhausted window resets.
        """
        if not self.config.o_enabled:
            return LimitDecision(True)
        now = self._clock()
        minute_window = int(now // MINUTE_SECONDS)
        day_window = int(now // DAY_SECONDS)
        day_count = self._increment(f"rl:d:{day_window}:{client_key}", DAY_SECONDS)
        if day_count > self.config.per_day:
            return LimitDecision(False, int((day_window + 1) * DAY_SECONDS - now) + 1, "day")
        minute_count = self._increment(f"rl:m:{minute_window}:{client_key}", MINUTE_SECONDS)
        if minute_count > self.config.per_minute:
            return LimitDecision(
                False, int((minute_window + 1) * MINUTE_SECONDS - now) + 1, "minute"
            )
        return LimitDecision(True)
