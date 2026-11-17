"""Loghub HDFS_v1 (CC BY 4.0, Zenodo record 8196385): download, stream and labels.

Xu et al., "Detecting Large-Scale System Problems by Mining Console Logs", SOSP 2009.
Zhu et al., "Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics",
ISSRE 2023.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

HDFS_URL = "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"
HDFS_SHA256 = "04f919f2185821f23f045dca611a7586429bdabc601bd7b43f30005f8e289b01"
HDFS_MB = 186.6


def cache_dir() -> Path:
    path = Path(os.environ.get("LOGSENTINEL_CACHE", Path.home() / ".cache" / "logsentinel"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_hdfs() -> Path:
    target = cache_dir() / "HDFS_v1.zip"
    if target.exists() and _sha256(target) == HDFS_SHA256:
        return target
    print(f"downloading Loghub HDFS_v1 ({HDFS_MB} MB, CC BY 4.0) from Zenodo ...")
    tmp = target.with_suffix(".part")
    with urllib.request.urlopen(HDFS_URL, timeout=120) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    if _sha256(tmp) != HDFS_SHA256:
        tmp.unlink()
        raise RuntimeError("HDFS_v1.zip checksum mismatch")
    tmp.replace(target)
    return target


def iter_hdfs_lines(zip_path: Path, limit: int | None = None) -> Iterator[str]:
    with zipfile.ZipFile(zip_path) as zf, zf.open("HDFS.log") as fh:
        for i, line in enumerate(io.TextIOWrapper(fh, encoding="utf-8", errors="replace")):
            if limit is not None and i >= limit:
                return
            yield line


def hdfs_labels(zip_path: Path) -> dict[str, int]:
    with zipfile.ZipFile(zip_path) as zf:
        text = zf.read("preprocessed/anomaly_label.csv").decode()
    labels = {}
    for line in text.splitlines()[1:]:
        blk, _, lab = line.partition(",")
        labels[blk] = 1 if lab.strip() == "Anomaly" else 0
    return labels


def iter_text_lines(path: Path) -> Iterator[str]:
    """Plain or gzip-compressed log file."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        yield from fh
