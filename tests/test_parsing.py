from __future__ import annotations

import numpy as np
from conftest import synthetic_log

from logsentinel import sessions
from logsentinel.parser import UNKNOWN_ID, TemplateMiner, mask, parse_hdfs_line


def test_parse_hdfs_line_fields() -> None:
    rec = parse_hdfs_line(
        "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block "
        "blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010\n"
    )
    assert rec is not None and rec.level == "INFO"
    assert rec.component == "dfs.DataNode$DataXceiver"
    assert rec.blocks == ["blk_-1608999687919862906"]
    assert rec.timestamp is not None and rec.timestamp.year == 2008
    assert parse_hdfs_line("garbage") is None


def test_mask_replaces_variables() -> None:
    assert mask("Received block blk_-42 of size 67108864 from /10.1.2.3:50010") == (
        "Received block <BLK> of size <NUM> from <IP>"
    )
    assert mask("Deleting block blk_1 file /mnt/hadoop/dfs/data/current/subdir42/blk_1") == (
        "Deleting block <BLK> file <PATH><NUM>/<BLK>"
    )


def test_miner_merges_variants_and_matches_unseen() -> None:
    m = TemplateMiner()
    a = m.learn("comp allocateBlock /a/x <BLK>")
    b = m.learn("comp allocateBlock /b/y <BLK>")
    assert a == b and m.templates[0].text == "comp allocateBlock <*> <BLK>"
    assert m.match("comp allocateBlock /zzz <BLK>") == a
    assert m.match("totally different message here") == UNKNOWN_ID


def test_miner_roundtrip(tmp_path) -> None:
    m = TemplateMiner()
    for k in ["x Receiving block <BLK>", "x Served block <BLK> to <IP>"]:
        m.learn(k)
    m.save(tmp_path / "t.json")
    loaded = TemplateMiner.load(tmp_path / "t.json")
    assert loaded.feature_names() == m.feature_names()
    assert loaded.match("x Served block <BLK> to <IP>") == 2


def test_scan_and_count_matrix() -> None:
    lines, labels = synthetic_log(50, 5)
    raw = sessions.scan(lines)
    assert set(raw.block_ids) == set(labels) and raw.n_lines == len(lines)
    miner = sessions.mine_templates(raw)
    x = sessions.count_matrix(raw, sessions.key_to_template(raw, miner), miner.n_features)
    assert x.shape == (55, miner.n_features)
    assert x.sum() == len(lines)  # each line mentions exactly one block here
    n_lines = {b: 5 for b in labels if not labels[b]} | {b: 3 for b in labels if labels[b]}
    assert np.array_equal(x.sum(1), [n_lines[b] for b in raw.block_ids])


def test_templates_from_train_period_only() -> None:
    lines, _ = synthetic_log(50, 5)
    raw = sessions.scan(lines)
    first_anomaly_line = next(i for i, ln in enumerate(lines) if "exception" in ln)
    miner = sessions.mine_templates(raw, max_line=first_anomaly_line)
    tid = sessions.key_to_template(raw, miner)
    assert (tid == UNKNOWN_ID).sum() == 1  # the exception template was never seen in training
