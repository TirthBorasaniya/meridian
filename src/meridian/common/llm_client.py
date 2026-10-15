"""LLM access through one caching client: every call is keyed by the SHA-256 of the model, the
messages, and the sampling parameters, so repeated evaluation runs and judge calls are
deterministic and free.

The transport is Groq's chat completions API; tests substitute a scripted transport. Token
usage is recorded per call into the active usage scopes and an optional metrics hook.
"""

import contextvars
import hashlib
import json
import logging
import random
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from meridian.common.config import ModelsConfig, RoleConfig

logger = logging.getLogger(__name__)

# ============= Constants =============

CACHE_SCHEMA_VERSION = 1
JSON_FENCE_CHARS = "`"
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


# ============= Responses and usage =============


@dataclass
class LLMResponse:
    """One completion with its accounting."""

    text: str
    model: str
    role: str
    input_tokens: int
    output_tokens: int
    o_cached: bool
    latency_s: float


@dataclass
class UsageMeter:
    """Accumulates calls and tokens per model within a scope."""

    call_list: list[LLMResponse] = field(default_factory=list)

    def record(self, response: LLMResponse) -> None:
        """Add one call."""
        self.call_list.append(response)

    @property
    def calls(self) -> int:
        """Number of calls, cached or not."""
        return len(self.call_list)

    @property
    def uncached_calls(self) -> int:
        """Number of calls that reached the provider."""
        return sum(1 for response in self.call_list if not response.o_cached)

    def tokens_by_model(self, o_include_cached: bool = True) -> dict[str, dict[str, int]]:
        """Return input and output token totals per model."""
        total_dict: dict[str, dict[str, int]] = {}
        for response in self.call_list:
            if response.o_cached and not o_include_cached:
                continue
            entry = total_dict.setdefault(response.model, {"input": 0, "output": 0, "calls": 0})
            entry["input"] += response.input_tokens
            entry["output"] += response.output_tokens
            entry["calls"] += 1
        return total_dict

    def calls_by_role(self) -> dict[str, int]:
        """Return the number of calls per role."""
        count_dict: dict[str, int] = {}
        for response in self.call_list:
            count_dict[response.role] = count_dict.get(response.role, 0) + 1
        return count_dict

    def cost_usd(self, models_config: ModelsConfig, o_include_cached: bool = True) -> float:
        """
        Convert token totals into USD with the configured per-million-token prices.

        Parameters
        ----------
        models_config : ModelsConfig
            Supplies `prices_usd_per_million_tokens`.
        o_include_cached : bool
            Count cached calls at the price they would have cost uncached.

        Returns
        -------
        cost : float
            Total cost in USD; models without a configured price raise KeyError.
        """
        cost = 0.0
        for model, entry in self.tokens_by_model(o_include_cached).items():
            price = models_config.prices_usd_per_million_tokens[model]
            cost += entry["input"] * price.input / 1e6 + entry["output"] * price.output / 1e6
        return cost


_ACTIVE_METERS: contextvars.ContextVar[tuple[UsageMeter, ...]] = contextvars.ContextVar(
    "active_usage_meters", default=()
)


@contextmanager
def usage_scope() -> Iterator[UsageMeter]:
    """Collect every LLM call made inside the `with` block, including nested scopes."""
    meter = UsageMeter()
    token = _ACTIVE_METERS.set(_ACTIVE_METERS.get() + (meter,))
    try:
        yield meter
    finally:
        _ACTIVE_METERS.reset(token)


# ============= Cache =============


def cache_key(model: str, message_list: list[dict], param_dict: dict) -> str:
    """Return the SHA-256 of the canonical JSON of model, messages, and parameters."""
    canonical = json.dumps(
        {"v": CACHE_SCHEMA_VERSION, "model": model, "messages": message_list, "params": param_dict},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache(Protocol):
    """Storage for completed responses keyed by `cache_key`."""

    def get(self, key: str) -> dict | None: ...

    def set(self, key: str, value: dict) -> None: ...


class MemoryResponseCache:
    """In-process response cache."""

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def get(self, key: str) -> dict | None:
        """Return a cached response."""
        return self._store.get(key)

    def set(self, key: str, value: dict) -> None:
        """Store a response."""
        self._store[key] = value


class SqliteResponseCache:
    """Response cache in a local SQLite file, safe across threads."""

    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._connection.commit()

    def get(self, key: str) -> dict | None:
        """Return a cached response."""
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict) -> None:
        """Store a response."""
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
            self._connection.commit()


class RedisResponseCache:
    """Response cache in Redis with an optional TTL."""

    def __init__(self, redis_client, prefix: str = "llm:", ttl_seconds: int | None = None) -> None:
        self._redis = redis_client
        self._prefix = prefix
        self._ttl_seconds = ttl_seconds

    def get(self, key: str) -> dict | None:
        """Return a cached response."""
        raw = self._redis.get(f"{self._prefix}{key}")
        return json.loads(raw) if raw else None

    def set(self, key: str, value: dict) -> None:
        """Store a response."""
        self._redis.set(f"{self._prefix}{key}", json.dumps(value), ex=self._ttl_seconds)


# ============= Transport =============


@dataclass
class Completion:
    """Raw provider output."""

    text: str
    input_tokens: int
    output_tokens: int


class Transport(Protocol):
    """Provider call: returns the completion, streaming deltas to `on_delta` when given."""

    def complete(
        self,
        model: str,
        message_list: list[dict],
        param_dict: dict,
        on_delta: Callable[[str], None] | None,
    ) -> Completion: ...


class RetryableError(Exception):
    """A transient provider failure, carrying an optional server-requested delay."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class GroqTransport:
    """Groq chat completions; maps transient SDK errors to `RetryableError`."""

    def __init__(self, api_key: str, timeout_s: float) -> None:
        import groq

        self._groq = groq
        self._client = groq.Groq(api_key=api_key, timeout=timeout_s, max_retries=0)

    @staticmethod
    def _retry_after(exc) -> float | None:
        response = getattr(exc, "response", None)
        header = response.headers.get("retry-after") if response is not None else None
        try:
            return float(header) if header is not None else None
        except ValueError:
            return None

    def complete(
        self,
        model: str,
        message_list: list[dict],
        param_dict: dict,
        on_delta: Callable[[str], None] | None,
    ) -> Completion:
        """Call the provider once."""
        request = dict(param_dict)
        if request.pop("json_mode", False):
            request["response_format"] = {"type": "json_object"}
        try:
            if on_delta is None:
                completion = self._client.chat.completions.create(
                    model=model, messages=cast(Any, message_list), **request
                )
                usage = completion.usage
                return Completion(
                    text=completion.choices[0].message.content or "",
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                )
            # Groq reports streamed usage on the final chunk under x_groq
            stream = self._client.chat.completions.create(
                model=model, messages=cast(Any, message_list), stream=True, **request
            )
            part_list: list[str] = []
            input_tokens = output_tokens = 0
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    part_list.append(delta)
                    on_delta(delta)
                usage = getattr(chunk, "usage", None) or getattr(
                    getattr(chunk, "x_groq", None), "usage", None
                )
                if usage is not None:
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
            return Completion("".join(part_list), input_tokens, output_tokens)
        except self._groq.RateLimitError as exc:
            raise RetryableError(str(exc), self._retry_after(exc)) from exc
        except (self._groq.InternalServerError, self._groq.APIConnectionError) as exc:
            raise RetryableError(str(exc)) from exc


# ============= Client =============


def strip_reasoning(text: str) -> str:
    """Remove inline `<think>...</think>` reasoning, which some models put in the content."""
    return THINK_BLOCK_PATTERN.sub("", text).strip()


def strip_json_fences(text: str) -> str:
    """Remove Markdown code fences around a JSON body."""
    stripped = text.strip()
    if stripped.startswith(JSON_FENCE_CHARS * 3):
        stripped = stripped.strip(JSON_FENCE_CHARS)
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    return stripped.strip()


class LLMClient:
    """Role-based chat client with caching, retries, and usage accounting."""

    def __init__(
        self,
        models_config: ModelsConfig,
        transport: Transport,
        cache: ResponseCache | None = None,
        usage_hook: Callable[[LLMResponse], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.models_config = models_config
        self.transport = transport
        self.cache = cache
        self.usage_hook = usage_hook
        self._sleep = sleep

    def params_for(self, role_config: RoleConfig, json_mode: bool, override_dict: dict) -> dict:
        """Return the sampling parameters sent to the provider and hashed into the key."""
        param_dict: dict[str, Any] = {
            "temperature": role_config.temperature,
            "max_tokens": role_config.max_tokens,
        }
        if role_config.reasoning_effort is not None:
            param_dict["reasoning_effort"] = role_config.reasoning_effort
        if json_mode:
            param_dict["json_mode"] = True
        param_dict.update(override_dict)
        return param_dict

    def _complete_with_retries(
        self,
        model: str,
        message_list: list[dict],
        param_dict: dict,
        on_delta: Callable[[str], None] | None,
    ) -> Completion:
        config = self.models_config
        for attempt in range(config.max_retries + 1):
            try:
                return self.transport.complete(model, message_list, param_dict, on_delta)
            except RetryableError as exc:
                if attempt >= config.max_retries:
                    raise
                backoff = min(
                    config.backoff_max_seconds, config.backoff_base_seconds * (2**attempt)
                )
                delay = max(exc.retry_after_s or 0.0, backoff) + random.uniform(0.0, 0.5)
                logger.warning(
                    "llm call retried",
                    extra={"model": model, "attempt": attempt + 1, "delay_s": round(delay, 2)},
                )
                self._sleep(delay)
        raise RuntimeError("unreachable")

    def chat(
        self,
        role: str,
        message_list: list[dict],
        json_mode: bool = False,
        on_delta: Callable[[str], None] | None = None,
        o_use_cache: bool = True,
        **override_dict: Any,
    ) -> LLMResponse:
        """
        Complete a chat for one role.

        Parameters
        ----------
        role : str
            A role in configs/models.yaml, for example `generator`.
        message_list : list of dict
            OpenAI-style messages with `role` and `content`.
        json_mode : bool
            Ask the provider for a JSON object.
        on_delta : callable or None
            Receives streamed text deltas; a cached response is delivered as one delta.
        o_use_cache : bool
            Read and write the response cache.
        **override_dict
            Parameter overrides, for example `temperature`.

        Returns
        -------
        response : LLMResponse
            Text and accounting.
        """
        role_config = self.models_config.role(role)
        param_dict = self.params_for(role_config, json_mode, override_dict)
        key = cache_key(role_config.model, message_list, param_dict)
        started = time.perf_counter()
        cached = self.cache.get(key) if (self.cache is not None and o_use_cache) else None
        if cached is not None:
            if on_delta is not None:
                on_delta(cached["text"])
            response = LLMResponse(
                text=cached["text"],
                model=role_config.model,
                role=role,
                input_tokens=cached["input_tokens"],
                output_tokens=cached["output_tokens"],
                o_cached=True,
                latency_s=time.perf_counter() - started,
            )
        else:
            completion = self._complete_with_retries(
                role_config.model, message_list, param_dict, on_delta
            )
            completion.text = strip_reasoning(completion.text)
            response = LLMResponse(
                text=completion.text,
                model=role_config.model,
                role=role,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                o_cached=False,
                latency_s=time.perf_counter() - started,
            )
            if self.cache is not None and o_use_cache:
                self.cache.set(
                    key,
                    {
                        "text": completion.text,
                        "input_tokens": completion.input_tokens,
                        "output_tokens": completion.output_tokens,
                    },
                )
        for meter in _ACTIVE_METERS.get():
            meter.record(response)
        if self.usage_hook is not None:
            self.usage_hook(response)
        return response

    def chat_json(self, role: str, message_list: list[dict], **override_dict: Any) -> dict:
        """
        Complete a chat in JSON mode and parse the object.

        Raises
        ------
        ValueError
            If the response is not a JSON object.
        """
        response = self.chat(role, message_list, json_mode=True, **override_dict)
        try:
            parsed = json.loads(strip_json_fences(response.text))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{role} returned invalid JSON: {response.text[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{role} returned JSON that is not an object")
        return parsed


# ============= Construction =============


def build_llm_client(
    models_config: ModelsConfig | None = None,
    cache: ResponseCache | None = None,
    usage_hook: Callable[[LLMResponse], None] | None = None,
) -> LLMClient:
    """
    Build the Groq-backed client from settings.

    Parameters
    ----------
    models_config : ModelsConfig or None
        Defaults to configs/models.yaml.
    cache : ResponseCache or None
        Defaults to the SQLite cache at `LLM_CACHE_PATH`.
    usage_hook : callable or None
        Called after every call, for metrics.

    Returns
    -------
    client : LLMClient
        Ready client.

    Raises
    ------
    RuntimeError
        If `GROQ_API_KEY` is not set.
    """
    from meridian.common.config import get_models_config
    from meridian.common.settings import get_settings

    settings = get_settings()
    config = models_config or get_models_config()
    api_key = settings.groq_api_key.get_secret_value()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    transport = GroqTransport(api_key, config.request_timeout_seconds)
    return LLMClient(
        config,
        transport,
        cache if cache is not None else SqliteResponseCache(settings.llm_cache_path),
        usage_hook,
    )
