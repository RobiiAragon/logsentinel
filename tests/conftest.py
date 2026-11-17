from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

NORMAL = [
    "INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/job_{j}/part-{j}. {b}",
    "INFO dfs.DataNode$DataXceiver: Receiving block {b} src: /10.0.0.{i}:5{i}0 dest: /10.0.0.{i}:50010",
    "INFO dfs.DataNode$PacketResponder: Received block {b} of size 6710 from /10.0.0.{i}",
    "INFO dfs.DataNode$PacketResponder: PacketResponder {i} for block {b} terminating",
    "INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.0.0.{i}:50010 is added to {b} size 6710",
]
ANOMALY = "WARN dfs.DataNode$DataXceiver: writeBlock {b} received exception java.io.IOException: Could not read from stream"


def synthetic_log(
    n_normal: int = 300, n_anomaly: int = 20, seed: int = 0
) -> tuple[list[str], dict[str, int]]:
    """HDFS-format lines: normal blocks follow the write pipeline, anomalous blocks stop
    early and raise an exception (the same shape as real HDFS failures)."""
    rng = np.random.default_rng(seed)
    lines, labels = [], {}
    for k in range(n_normal + n_anomaly):
        blk = f"blk_{1000 + k}"
        anomalous = k >= n_normal
        labels[blk] = int(anomalous)
        steps = [*NORMAL[:2], ANOMALY] if anomalous else NORMAL
        for s, tpl in enumerate(steps):
            i = int(rng.integers(1, 9))
            ts = f"081109 20{(k // 60) % 60:02d}{(k + s) % 60:02d}"
            lines.append(f"{ts} {100 + i} " + tpl.format(b=blk, i=i, j=k) + "\n")
    order = rng.permutation(len(lines))  # interleave like a real cluster log
    return [lines[i] for i in sorted(order[: len(lines)])], labels


@pytest.fixture()
def synthetic(tmp_path: Path) -> tuple[Path, Path, dict[str, int]]:
    lines, labels = synthetic_log()
    log = tmp_path / "syn.log"
    log.write_text("".join(lines))
    lab = tmp_path / "labels.csv"
    lab.write_text(
        "BlockId,Label\n"
        + "".join(f"{b},{'Anomaly' if v else 'Normal'}\n" for b, v in labels.items())
    )
    return log, lab, labels
