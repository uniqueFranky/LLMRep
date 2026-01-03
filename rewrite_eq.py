#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rewrite YokyYao/Diversity_Challenge (eq) questions into stricter "enumeration prompts"
using DeepSeek (OpenAI-compatible API).

Input: HF dataset row: {"question": "..."}
Output JSONL lines: {"id": int, "question": original, "rewritten_question": rewritten}

Usage:
  export DEEPSEEK_API_KEY="..."
  # optional: export DEEPSEEK_BASE_URL="https://api.deepseek.com"  (example)
  python rewrite_eq_with_deepseek.py --out rewritten_eq_dataset.jsonl --concurrency 16
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from typing import Dict, Any, List, Optional

from datasets import load_dataset

# If you don't have it:
#   pip install openai datasets
from openai import AsyncOpenAI


REWRITE_SPEC = r"""
You are rewriting a dataset prompt to make it suitable for evaluating repetition in LLM generation.

Your goal is NOT to use a fixed template.
Your goal is to produce a clear enumeration instruction while preserving linguistic diversity.

Rewrite rules (must follow all):

1) The rewritten prompt MUST explicitly require a specific NUMBER of items.
   - The number does NOT need to be 50.
   - Choose a reasonable number (20 ~ 80) based on the original task.
   - The number must be explicit and unambiguous.

2) The prompt MUST clearly instruct the model to enumerate multiple items.
   - You do NOT have to use the word "list".
   - Any phrasing that clearly means enumeration is acceptable
     (e.g., generate, provide, give, name, produce, write out).

3) The prompt MUST require that the output has:
   - no index
   - no newline
   - Do NOT put the constraint at the end of the rewritten prompt because you must place the example items at the end.
   (exact wording is flexible, but the constraint must be clear).

4) If the original prompt contains example items or partial enumerations,
   you MUST preserve them and place them AT THE END of the rewritten prompt.
   - You do NOT have to use "such as".
   - Any equivalent phrasing is allowed
     (e.g., including, for example, like, starting with).

5) The rewritten prompt MUST NOT end with a period ".".

6) The rewritten prompt MUST be a single line (no newline characters).

7) Do NOT add explanations, meta-comments, or formatting hints.
   Output ONLY the rewritten prompt.

Important:
- Vary sentence structures across different prompts.
- Avoid using the same phrasing repeatedly.
- Preserve the original semantic intent; do not change the task category.

For example:
before: Please write down as many names as you beginning with letter 'A': Alice, Ann, Andrew
after: Please write down 50 names beginning with letter 'A' without index and newline, such as Alice, Ann, Andrew
"""


def normalize_one_line(s: str) -> str:
    # Force single line, remove leading/trailing whitespace
    s = s.replace("\r", " ").replace("\n", " ").strip()
    # Remove ending period(s)
    s = re.sub(r"\.+\s*$", "", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s


def load_done_ids(out_path: str) -> set:
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(int(obj["id"]))
            except Exception:
                # ignore broken lines
                pass
    return done


async def rewrite_one(
    client: AsyncOpenAI,
    model: str,
    q: str,
    max_retries: int = 5,
) -> str:
    q = normalize_one_line(q)

    # A little randomness helps avoid synchronized retry storms
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": REWRITE_SPEC.strip()},
                    {"role": "user", "content": f"Original prompt: {q}"},
                ],
            )
            out = resp.choices[0].message.content or ""
            out = normalize_one_line(out)

            # Minimal safety checks to enforce your constraints
            # Ensure it contains the required phrase
            if "without index and newline" not in out.lower():
                # If model forgot, patch it in a conservative way
                out = re.sub(
                    r"\bwithout\b.*?\bnewline\b",
                    "without index and newline",
                    out,
                    flags=re.IGNORECASE,
                )
                if "without index and newline" not in out.lower():
                    # Insert before ", such as" if present else near the end
                    if ", such as" in out.lower():
                        out = re.sub(
                            r",\s*such as",
                            ", without index and newline, such as",
                            out,
                            flags=re.IGNORECASE,
                        )
                    else:
                        out = out + " without index and newline"

            out = normalize_one_line(out)
            return out

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            backoff = (2 ** attempt) + random.random()
            await asyncio.sleep(backoff)


async def worker(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    item: Dict[str, Any],
    out_f,
):
    async with sem:
        rid = item["id"]
        original = item["question"]
        rewritten = await rewrite_one(client, model, original)
        rec = {"id": rid, "question": original, "rewritten_question": rewritten}
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()


async def main_async(args):
    # 1) Load dataset
    ds = load_dataset("YokyYao/Diversity_Challenge", split="train")  # has "question" column
    # dataset viewer shows 500 rows; we just iterate whatever is in split
    # (If HF changes, this still works.)
    # ds = ds.select(range(10))
    # 2) Resume
    done_ids = load_done_ids(args.out)
    print(f"[INFO] already done: {len(done_ids)} rows", file=sys.stderr)

    # 3) DeepSeek client (OpenAI-compatible)
    api_key = "sk-21a2cc2d84b14db983d1b1f26ab42450"
    if not api_key:
        raise RuntimeError("Missing env var DEEPSEEK_API_KEY")

    base_url = "https://api.deepseek.com"  # set if needed
    client = AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)

    sem = asyncio.Semaphore(args.concurrency)

    # 4) Schedule tasks
    tasks = []
    with open(args.out, "a", encoding="utf-8") as out_f:
        for i, row in enumerate(ds):
            if i in done_ids:
                continue
            item = {"id": i, "question": row["question"]}
            tasks.append(asyncio.create_task(worker(sem, client, args.model, item, out_f)))

        # 5) Progress (simple)
        total = len(tasks)
        print(f"[INFO] to process: {total}", file=sys.stderr)

        finished = 0
        for fut in asyncio.as_completed(tasks):
            await fut
            finished += 1
            if finished % max(1, args.log_every) == 0:
                print(f"[INFO] progress: {finished}/{total}", file=sys.stderr)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rewritten_eq_dataset.jsonl")
    ap.add_argument("--model", default="deepseek-chat")  # change if your DeepSeek deployment uses another name
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=20)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
