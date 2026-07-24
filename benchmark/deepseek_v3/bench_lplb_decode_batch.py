#!/usr/bin/env python3
"""Benchmark a DP-attention decode run with one batched /generate request.

Using one HTTP batch ensures that every request is dispatched before the
server waits for the first response. This avoids serial admission of native
/generate calls when EP workers must participate in the same DeepEP steps.
"""

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-len", type=int, default=8)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def make_input_ids(
    *, vocab_size: int, batch_size: int, input_len: int, seed: int
) -> list[list[int]]:
    random.seed(seed)
    np.random.seed(seed)
    offsets = np.random.randint(0, vocab_size, size=batch_size)
    return [
        [int((offsets[i] + i + j) % vocab_size) for j in range(input_len)]
        for i in range(batch_size)
    ]


def response_count(payload) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return len(payload["data"])
    return 1


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_ids = make_input_ids(
        vocab_size=tokenizer.vocab_size,
        batch_size=args.batch_size,
        input_len=args.input_len,
        seed=args.seed,
    )
    payload = {
        "input_ids": input_ids,
        "sampling_params": [
            {
                "temperature": 0.0,
                "max_new_tokens": args.output_len,
                "ignore_eos": True,
            }
            for _ in range(args.batch_size)
        ],
        "stream": False,
    }
    encoded_inputs = json.dumps(input_ids, separators=(",", ":")).encode()
    start_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    response = requests.post(
        f"http://{args.host}:{args.port}/generate",
        json=payload,
        timeout=6 * 60 * 60,
    )
    duration_s = time.perf_counter() - start
    end_utc = datetime.now(timezone.utc).isoformat()
    response.raise_for_status()
    response_payload = response.json()
    count = response_count(response_payload)
    if count != args.batch_size:
        raise RuntimeError(
            f"Expected {args.batch_size} responses from batch request, got {count}"
        )

    total_input_tokens = args.batch_size * args.input_len
    total_output_tokens = args.batch_size * args.output_len
    result = {
        "start_utc": start_utc,
        "end_utc": end_utc,
        "duration_s": duration_s,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "seed": args.seed,
        "vocab_size": tokenizer.vocab_size,
        "input_ids_sha256": hashlib.sha256(encoded_inputs).hexdigest(),
        "response_count": count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "output_throughput_tok_s": total_output_tokens / duration_s,
        "total_throughput_tok_s": (total_input_tokens + total_output_tokens)
        / duration_s,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
