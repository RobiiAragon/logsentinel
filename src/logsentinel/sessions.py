"""Group log lines into per-block sessions and build event-count matrices."""

from __future__ import annotations

import time
from array import array
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .parser import UNKNOWN_ID, TemplateMiner, parse_hdfs_line


@dataclass
class RawSessions:
    """Result of one streaming pass: everything needed to build features later."""

    block_ids: list[str]
    first_line: np.ndarray  # first line index where each block appears (time order)
    first_ts: np.ndarray  # unix seconds of first appearance (0 if unparsable)
    pair_block: np.ndarray  # (line, block) pairs -> block index
    pair_key: np.ndarray  # (line, block) pairs -> unique masked-key index
    keys: list[str]  # unique masked keys
    key_count: np.ndarray  # occurrences of each key
    key_first_line: np.ndarray  # first line index of each key
    n_lines: int
    seconds: float


def scan(lines: Iterable[str]) -> RawSessions:
    """Single pass over raw HDFS lines. Memory ~ 8 bytes per (line, block) pair."""
    t0 = time.perf_counter()
    block_index: dict[str, int] = {}
    key_index: dict[str, int] = {}
    keys: list[str] = []
    key_count = array("q")
    key_first = array("q")
    first_line = array("q")
    first_ts = array("q")
    pair_block = array("i")
    pair_key = array("i")
    n = -1
    for n, raw in enumerate(lines):
        rec = parse_hdfs_line(raw)
        if rec is None or not rec.blocks:
            continue
        k = rec.key
        kid = key_index.get(k)
        if kid is None:
            kid = key_index[k] = len(keys)
            keys.append(k)
            key_count.append(0)
            key_first.append(n)
        key_count[kid] += 1
        for blk in dict.fromkeys(rec.blocks):  # unique, order-preserving
            bid = block_index.get(blk)
            if bid is None:
                bid = block_index[blk] = len(first_line)
                first_line.append(n)
                first_ts.append(int(rec.timestamp.timestamp()) if rec.timestamp else 0)
            pair_block.append(bid)
            pair_key.append(kid)
    return RawSessions(
        list(block_index),
        np.frombuffer(first_line, np.int64).copy(),
        np.frombuffer(first_ts, np.int64).copy(),
        np.frombuffer(pair_block, np.int32).copy(),
        np.frombuffer(pair_key, np.int32).copy(),
        keys,
        np.frombuffer(key_count, np.int64).copy(),
        np.frombuffer(key_first, np.int64).copy(),
        n + 1,
        time.perf_counter() - t0,
    )


def mine_templates(
    raw: RawSessions, max_line: int | None = None, sim_threshold: float = 0.5
) -> TemplateMiner:
    """Learn templates only from keys first seen before ``max_line`` (no test leakage)."""
    miner = TemplateMiner(sim_threshold=sim_threshold)
    order = np.argsort(-raw.key_count, kind="stable")  # frequent keys first: stable templates
    for kid in order:
        if max_line is None or raw.key_first_line[kid] < max_line:
            miner.learn(raw.keys[kid], int(raw.key_count[kid]))
    return miner


def key_to_template(raw: RawSessions, miner: TemplateMiner) -> np.ndarray:
    return np.array([miner.match(k) for k in raw.keys], dtype=np.int32)


def count_matrix(raw: RawSessions, key_tid: np.ndarray, n_features: int) -> np.ndarray:
    counts = np.zeros((len(raw.block_ids), n_features), dtype=np.float32)
    np.add.at(counts, (raw.pair_block, key_tid[raw.pair_key]), 1.0)
    return counts


def counts_for_lines(lines: Iterable[str], miner: TemplateMiner) -> tuple[list[str], np.ndarray]:
    """Inference-time featurisation of an arbitrary HDFS log (unseen keys -> UNKNOWN)."""
    raw = scan(lines)
    tid = key_to_template(raw, miner)
    tid[tid < 0] = UNKNOWN_ID
    return raw.block_ids, count_matrix(raw, tid, miner.n_features)
