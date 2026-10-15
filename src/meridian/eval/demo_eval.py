"""E6: the public demo. Cold start (restart the Space, then time until `/readyz` answers 200),
p50 and p95 end-to-end latency of 20 scripted questions through `/v1/ask`, and rate-limit
behavior under a burst against `/v1/search`. Writes `results/e6_demo.json`.
"""

import json
import logging
import time

import httpx
from pydantic import Field

from meridian.common.config import VersionedConfig, load_config
from meridian.common.results import latency_summary, write_result
from meridian.common.settings import get_settings
from meridian.eval.datasets import load_answer_set

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/eval/e6.yaml"
POLL_INTERVAL_S = 5.0


class E6Config(VersionedConfig):
    """configs/eval/e6.yaml."""

    base_url: str
    questions_path: str
    questions_split: str
    question_interval_s: float = Field(ge=0.0)
    o_restart_for_cold_start: bool
    cold_start_timeout_s: float = Field(gt=0)
    rate_limit_burst: int = Field(gt=0)
    output: str


def measure_cold_start(base_url: str, space_repo: str, timeout_s: float) -> dict:
    """Restart the Space and time how long `/readyz` takes to return 200."""
    from huggingface_hub import HfApi

    token = get_settings().hf_token.get_secret_value()
    HfApi(token=token).restart_space(space_repo)
    started = time.perf_counter()
    first_status = None
    with httpx.Client(timeout=30) as client:
        while time.perf_counter() - started < timeout_s:
            try:
                response = client.get(f"{base_url}/readyz")
                first_status = first_status or response.status_code
                if response.status_code == 200:
                    return {"seconds": time.perf_counter() - started, "o_ready": True}
            except httpx.HTTPError as exc:
                logger.info("demo not reachable yet", extra={"error": str(exc)})
            time.sleep(POLL_INTERVAL_S)
    return {"seconds": None, "o_ready": False, "timeout_s": timeout_s}


def ask_once(client: httpx.Client, base_url: str, question: str) -> dict:
    """Send one question and read the SSE stream to the final event."""
    started = time.perf_counter()
    first_token_s = None
    final: dict = {}
    with client.stream("POST", f"{base_url}/v1/ask", json={"question": question}) as response:
        if response.status_code != 200:
            return {"status_code": response.status_code, "latency_s": time.perf_counter() - started}
        event = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                if event == "token" and first_token_s is None:
                    first_token_s = time.perf_counter() - started
                if event == "final":
                    final = json.loads(line[len("data: ") :])
    return {
        "status_code": 200,
        "latency_s": time.perf_counter() - started,
        "first_token_s": first_token_s,
        "status": final.get("status"),
        "cached": final.get("cached"),
        "citations": len(final.get("citations", [])),
    }


def measure_rate_limit(client: httpx.Client, base_url: str, burst: int) -> dict:
    """Fire a burst at `/v1/search` and record the status codes and Retry-After."""
    status_list: list[int] = []
    retry_after = None
    for _ in range(burst):
        response = client.post(f"{base_url}/v1/search", json={"query": "chain-of-thought", "k": 3})
        status_list.append(response.status_code)
        if response.status_code == 429 and retry_after is None:
            retry_after = response.headers.get("Retry-After")
    return {
        "burst": burst,
        "status_codes": status_list,
        "allowed": sum(1 for code in status_list if code == 200),
        "limited": sum(1 for code in status_list if code == 429),
        "first_429_index": status_list.index(429) if 429 in status_list else None,
        "retry_after_s": int(retry_after) if retry_after else None,
    }


def run_e6(config: E6Config) -> dict:
    """Run cold start, scripted questions, and the burst."""
    settings = get_settings()
    base_url = (config.base_url or settings.demo_url).rstrip("/")
    if not base_url:
        raise RuntimeError("set base_url in configs/eval/e6.yaml or MERIDIAN_DEMO_URL")
    cold_start = (
        measure_cold_start(base_url, settings.hf_space_repo, config.cold_start_timeout_s)
        if config.o_restart_for_cold_start
        else None
    )
    question_list = [
        item["question"] for item in load_answer_set(config.questions_path, config.questions_split)
    ]
    answer_list: list[dict] = []
    with httpx.Client(timeout=180) as client:
        index_info = client.get(f"{base_url}/v1/index").json()
        for question in question_list:
            answer_list.append(ask_once(client, base_url, question))
            time.sleep(config.question_interval_s)
        # a fresh minute window, so the burst measures the limit rather than earlier traffic
        time.sleep(61)
        rate_limit = measure_rate_limit(client, base_url, config.rate_limit_burst)
    ok_list = [answer for answer in answer_list if answer["status_code"] == 200]
    return {
        "base_url": base_url,
        "index": index_info,
        "cold_start": cold_start,
        "questions": len(question_list),
        "latency": latency_summary([answer["latency_s"] for answer in ok_list]),
        "first_token": latency_summary(
            [answer["first_token_s"] for answer in ok_list if answer["first_token_s"] is not None]
        ),
        "answers": answer_list,
        "rate_limit": rate_limit,
    }


def cli_eval(args) -> int:
    """`meridian eval demo`: run E6 and write its results file."""
    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_config(E6Config, config_path)
    metrics = run_e6(config)
    write_result(
        config.output,
        metrics,
        config_path,
        notes=(
            "Latency is client-side from request to the final SSE event and includes network "
            "time. Cold start is measured from a Space restart to the first 200 from /readyz."
        ),
    )
    print(
        json.dumps({key: metrics[key] for key in ("cold_start", "latency", "rate_limit")}, indent=2)
    )
    return 0
