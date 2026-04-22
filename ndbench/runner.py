"""
Dispatches model calls for every (model, condition, profile, query) cell and
appends responses to data/responses/cache.jsonl. Idempotent: skips any cell
whose key is already present with a non-null response. Rerun to retry failures.

Usage:
  python -m ndbench.runner --dry-run        # print plan, no API calls
  python -m ndbench.runner --limit 4        # run only N cells (smoke test)
  python -m ndbench.runner                  # full sweep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data" / "responses"
CACHE_FILE = DATA_DIR / "cache.jsonl"

MODELS = {
    "gpt-5-chat-latest": "openai",
    "claude-sonnet-4-6": "anthropic",
}

TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 2048
OPENAI_CONCURRENCY = 2
ANTHROPIC_CONCURRENCY = 4
SDK_MAX_RETRIES = 10


@dataclass
class Cell:
    model: str
    condition_id: str
    condition_label: str
    profile_id: str
    profile_label: str
    query_id: str
    query_domain: str
    query_text: str
    system_prompt: str | None

    def key(self) -> str:
        return f"{self.model}|{self.condition_id}|{self.profile_id}|{self.query_id}"


def render_prompt(template: str | None, profile: dict) -> str | None:
    if template is None:
        return None
    traits = profile["traits"]
    traits_str = "; ".join(traits) if isinstance(traits, list) else str(traits)
    return (
        template
        .replace("{{neurotype}}", profile["neurotype"])
        .replace("{{traits}}", traits_str)
        .replace("{{communication_preference}}", profile["communication_preference"])
        .replace("{{response_format}}", profile["response_format"])
        .replace("{{response_detail_level}}", profile["response_detail_level"])
        .replace("{{coaching_tone}}", profile["coaching_tone"])
        .replace("{{freetext}}", profile["freetext"])
    )


def build_cells(profiles: list, conditions: list, queries_by_domain: dict) -> list[Cell]:
    cells: list[Cell] = []
    for model in MODELS:
        for condition in conditions:
            for profile in profiles:
                rendered = render_prompt(condition.get("system_prompt"), profile)
                for domain, qlist in queries_by_domain.items():
                    for q in qlist:
                        cells.append(Cell(
                            model=model,
                            condition_id=condition["id"],
                            condition_label=condition["label"],
                            profile_id=profile["id"],
                            profile_label=profile["label"],
                            query_id=q["id"],
                            query_domain=domain,
                            query_text=q["text"],
                            system_prompt=rendered,
                        ))
    return cells


def load_cache_keys() -> set[str]:
    if not CACHE_FILE.exists():
        return set()
    keys: set[str] = set()
    with CACHE_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("response") is not None and not obj.get("error"):
                keys.add(f'{obj["model"]}|{obj["condition_id"]}|{obj["profile_id"]}|{obj["query_id"]}')
    return keys


async def call_openai(client: AsyncOpenAI, cell: Cell) -> tuple[str, dict]:
    messages: list[dict] = []
    if cell.system_prompt:
        messages.append({"role": "system", "content": cell.system_prompt})
    messages.append({"role": "user", "content": cell.query_text})
    resp = await client.chat.completions.create(
        model=cell.model,
        messages=messages,
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
    )
    content = resp.choices[0].message.content or ""
    usage = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }
    return content, usage


async def call_anthropic(client: AsyncAnthropic, cell: Cell) -> tuple[str, dict]:
    kwargs: dict = {
        "model": cell.model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": cell.query_text}],
    }
    if cell.system_prompt:
        kwargs["system"] = cell.system_prompt
    resp = await client.messages.create(**kwargs)
    content = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    return content, usage


async def run_cell(cell: Cell, openai_client: AsyncOpenAI, anthropic_client: AsyncAnthropic,
                   semaphores: dict, fh, lock: asyncio.Lock, progress: dict) -> None:
    provider = MODELS[cell.model]
    sem = semaphores[provider]
    async with sem:
        start = time.time()
        try:
            if provider == "openai":
                content, usage = await call_openai(openai_client, cell)
            else:
                content, usage = await call_anthropic(anthropic_client, cell)
            record = {
                **asdict(cell),
                "response": content,
                "usage": usage,
                "temperature": TEMPERATURE,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latency_s": round(time.time() - start, 3),
                "error": None,
            }
        except Exception as e:
            record = {
                **asdict(cell),
                "response": None,
                "usage": None,
                "temperature": TEMPERATURE,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latency_s": round(time.time() - start, 3),
                "error": f"{type(e).__name__}: {e}",
            }
        async with lock:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            progress["done"] += 1
            if record["error"]:
                progress["errors"] += 1
            if progress["done"] % 10 == 0 or progress["done"] == progress["total"]:
                err_suffix = f" ({progress['errors']} errors)" if progress["errors"] else ""
                print(f"[{progress['done']}/{progress['total']}] {cell.key()}{err_suffix}", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print plan, no API calls")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N pending cells")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    openai_key = os.getenv("OPENAI")
    anthropic_key = os.getenv("CLAUDE")
    if not openai_key or not anthropic_key:
        print("Missing OPENAI or CLAUDE in .env", file=sys.stderr)
        sys.exit(1)

    profiles = yaml.safe_load((CONFIG_DIR / "profiles.yaml").read_text())["profiles"]
    conditions = yaml.safe_load((CONFIG_DIR / "prompts.yaml").read_text())["conditions"]
    queries_by_domain = yaml.safe_load((CONFIG_DIR / "queries.yaml").read_text())["queries"]

    cells = build_cells(profiles, conditions, queries_by_domain)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    done = load_cache_keys()
    todo = [c for c in cells if c.key() not in done]

    print(f"Total cells:     {len(cells)}")
    print(f"Already cached:  {len(done)}")
    print(f"Pending:         {len(todo)}")

    if args.limit is not None:
        todo = todo[: args.limit]
        print(f"Running (limit): {len(todo)}")

    if args.dry_run:
        print("\nFirst 5 pending cells:")
        for c in todo[:5]:
            print(f"  {c.key()}  [{c.query_domain}]")
            sp = (c.system_prompt or "<none>").replace("\n", " ")[:100]
            print(f"    system: {sp}")
            print(f"    query:  {c.query_text[:100]}")
        return

    if not todo:
        print("Nothing to do.")
        return

    openai_client = AsyncOpenAI(api_key=openai_key, max_retries=SDK_MAX_RETRIES)
    anthropic_client = AsyncAnthropic(api_key=anthropic_key, max_retries=SDK_MAX_RETRIES)
    semaphores = {
        "openai": asyncio.Semaphore(OPENAI_CONCURRENCY),
        "anthropic": asyncio.Semaphore(ANTHROPIC_CONCURRENCY),
    }
    lock = asyncio.Lock()
    progress = {"done": 0, "errors": 0, "total": len(todo)}

    t0 = time.time()
    with CACHE_FILE.open("a") as fh:
        tasks = [run_cell(c, openai_client, anthropic_client, semaphores, fh, lock, progress) for c in todo]
        await asyncio.gather(*tasks)
    elapsed = time.time() - t0

    done_after = load_cache_keys()
    missing = [c for c in cells if c.key() not in done_after]
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Successful: {len(done_after)}/{len(cells)}")
    if missing:
        print(f"Still missing: {len(missing)} — rerun to retry")


if __name__ == "__main__":
    asyncio.run(main())
