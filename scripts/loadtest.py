"""Load test of `POST /v1/search` on a running API: throughput and latency under concurrency.

Retrieval only, so it costs no LLM tokens. Writes `results/loadtest.json`.

Usage::

    uv run python scripts/loadtest.py --config configs/eval/loadtest.yaml
"""

import argparse
import asyncio
import time

import httpx
from pydantic import Field

from meridian.common.config import VersionedConfig, load_config
from meridian.common.results import latency_summary, write_result
from meridian.eval.datasets import read_jsonl


class LoadTestConfig(VersionedConfig):
    """configs/eval/loadtest.yaml."""

    base_url: str
    concurrency: int = Field(gt=0)
    requests: int = Field(gt=0)
    k: int = Field(gt=0)
    mode: str
    questions_path: str
    output: str


async def run_load(config: LoadTestConfig) -> dict:
    """Send the configured number of requests with bounded concurrency."""
    question_list = [record["question"] for record in read_jsonl(config.questions_path)]
    semaphore = asyncio.Semaphore(config.concurrency)
    latency_list: list[float] = []
    status_count: dict[int, int] = {}

    async with httpx.AsyncClient(base_url=config.base_url, timeout=120) as client:

        async def one(index: int) -> None:
            async with semaphore:
                started = time.perf_counter()
                response = await client.post(
                    "/v1/search",
                    json={
                        "query": question_list[index % len(question_list)],
                        "k": config.k,
                        "mode": config.mode,
                    },
                )
                latency_list.append(time.perf_counter() - started)
                status_count[response.status_code] = status_count.get(response.status_code, 0) + 1

        wall_start = time.perf_counter()
        await asyncio.gather(*(one(index) for index in range(config.requests)))
        wall_s = time.perf_counter() - wall_start
    return {
        "base_url": config.base_url,
        "concurrency": config.concurrency,
        "requests": config.requests,
        "wall_s": wall_s,
        "throughput_rps": config.requests / wall_s,
        "latency": latency_summary(latency_list),
        "status_codes": {str(code): count for code, count in sorted(status_count.items())},
    }


def main() -> None:
    """Run the load test and write its results file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/eval/loadtest.yaml")
    args = parser.parse_args()
    config = load_config(LoadTestConfig, args.config)
    metrics = asyncio.run(run_load(config))
    write_result(
        config.output,
        metrics,
        args.config,
        notes="Client-side latency of /v1/search including network; rate limiting must be off.",
    )
    print(metrics)


if __name__ == "__main__":
    main()
