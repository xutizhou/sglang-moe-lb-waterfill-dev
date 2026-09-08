#!/usr/bin/env python3
"""Fixed-shape decode benchmark with real prompts, pinned per DP rank.

Reads ``[sequences, tokens]`` token ids (e.g. the teacher-forced route dumps'
``input_ids.npy``), takes the first ``--num-seqs`` sequences and the first
``--prompt-len`` tokens of each as prompts, and issues one batched
``/generate`` request per DP rank with ``routed_dp_rank`` so every rank holds
exactly ``num_seqs / dp_size`` requests for the whole run (the same shape the
random-id harness produces through round-robin, but deterministic).  Greedy,
``ignore_eos`` and a fixed ``max_new_tokens`` make every run generate the same
number of tokens; the reported number is wall time for the whole batch.

The output log mirrors the fields ``summarize_graph_ab.py`` parses from
``sglang.benchmark.serving`` so both benchmark modes summarise identically.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time

import numpy as np
import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--input-ids-npy", required=True)
    parser.add_argument("--num-seqs", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--dp-size", type=int, default=4)
    parser.add_argument("--seq-offset", type=int, default=0, help="first sequence row to use")
    parser.add_argument(
        "--mode",
        choices=("pinned", "individual"),
        default="individual",
        help="pinned: one batched request per DP rank with routed_dp_rank; "
        "individual: num_seqs concurrent single-prompt requests, framework round-robin "
        "(the pattern sglang.benchmark.serving uses; pinned batches raced the DP "
        "admission on the first request in some runs)",
    )
    parser.add_argument("--label", default="real")
    parser.add_argument("--result-file")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    ids = np.load(args.input_ids_npy)
    if ids.ndim != 2:
        raise ValueError(f"{args.input_ids_npy}: expected [seq, tokens], got {ids.shape}")
    if args.seq_offset + args.num_seqs > ids.shape[0] or args.prompt_len > ids.shape[1]:
        raise ValueError(f"not enough sequences/tokens in {ids.shape} for the requested shape")
    if args.num_seqs % args.dp_size:
        raise ValueError("--num-seqs must be divisible by --dp-size")
    per_rank = args.num_seqs // args.dp_size
    prompts = ids[args.seq_offset : args.seq_offset + args.num_seqs, : args.prompt_len].astype(int).tolist()

    url = f"http://{args.host}:{args.port}/generate"
    sampling = {"temperature": 0.0, "max_new_tokens": args.output_len, "ignore_eos": True}

    def post(payload: dict, key: int) -> dict:
        start = time.perf_counter()
        response = requests.post(url, json=payload, timeout=args.timeout)
        response.raise_for_status()
        body = response.json()
        items = body if isinstance(body, list) else [body]
        return {
            "rank": key,
            "seconds": time.perf_counter() - start,
            "responses": len(items),
            "completion_tokens": sum(i.get("meta_info", {}).get("completion_tokens", 0) for i in items),
        }

    if args.mode == "pinned":
        barrier = threading.Barrier(args.dp_size)

        def one(rank: int) -> dict:
            payload = {
                "input_ids": prompts[rank * per_rank : (rank + 1) * per_rank],
                "routed_dp_rank": rank,
                "sampling_params": sampling,
                "stream": False,
            }
            barrier.wait()
            return post(payload, rank)

        workers, keys = args.dp_size, range(args.dp_size)
    else:
        barrier = threading.Barrier(args.num_seqs)

        def one(index: int) -> dict:
            payload = {"input_ids": prompts[index], "sampling_params": sampling, "stream": False}
            barrier.wait()
            return post(payload, index)

        workers, keys = args.num_seqs, range(args.num_seqs)

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        per_rank_results = list(pool.map(one, keys))
    wall = time.perf_counter() - start
    generated = sum(r["completion_tokens"] for r in per_rank_results)
    successful = sum(r["responses"] for r in per_rank_results)
    expected = args.num_seqs * args.output_len

    # Same field names as sglang.benchmark.serving so summarize_graph_ab.py works.
    print(f"Successful requests:                     {successful}")
    print(f"Benchmark duration (s):                  {wall:.3f}")
    print(f"Total generated tokens:                  {generated}")
    print(f"Output token throughput (tok/s):         {generated / wall:.2f}")
    print(f"Expected generated tokens:               {expected}")
    if args.mode == "pinned":
        for r in per_rank_results:
            print(f"rank {r['rank']}: {r['responses']} responses, {r['completion_tokens']} tokens, {r['seconds']:.3f} s")
    else:
        secs = [r["seconds"] for r in per_rank_results]
        print(f"per-request seconds: min {min(secs):.3f} max {max(secs):.3f}")

    result = {
        "label": args.label,
        "input_ids_npy": args.input_ids_npy,
        "num_seqs": args.num_seqs,
        "prompt_len": args.prompt_len,
        "output_len": args.output_len,
        "dp_size": args.dp_size,
        "mode": args.mode,
        "wall_seconds": wall,
        "generated_tokens": generated,
        "expected_tokens": expected,
        "output_tok_per_second": generated / wall,
        "per_rank": per_rank_results,
    }
    if args.result_file:
        with open(args.result_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
