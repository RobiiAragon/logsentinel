"""Log parsing: variable masking + a Drain-style template miner.

Masking (block ids, IPs, numbers, paths) collapses millions of lines into a few hundred
unique strings; the template miner then merges near-duplicates. Mining only ever runs
on unique strings, so parsing cost is dominated by one regex pass per line.

Reference: He et al., "Drain: An Online Log Parsing Approach with Fixed Depth Tree", ICWS 2017.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

BLOCK_RE = re.compile(r"blk_-?\d+")
MASKS: list[tuple[re.Pattern[str], str]] = [
    (BLOCK_RE, "<BLK>"),
    (re.compile(r"/?\d+\.\d+\.\d+\.\d+(?::\d+)?"), "<IP>"),
    (re.compile(r"(?<![A-Za-z])-?\d+(?![A-Za-z])"), "<NUM>"),
    (re.compile(r"(?:/[\w.\-]+)+"), "<PATH>"),
]
WILDCARD = "<*>"
UNKNOWN_ID = 0  # template id reserved for "never seen during training"


def mask(content: str) -> str:
    for rx, rep in MASKS:
        content = rx.sub(rep, content)
    return content


@dataclass
class HdfsLine:
    timestamp: datetime | None
    level: str
    component: str
    content: str
    blocks: list[str]

    @property
    def key(self) -> str:
        """What the miner sees: component + masked content."""
        return f"{self.component} {mask(self.content)}"


def parse_hdfs_line(raw: str) -> HdfsLine | None:
    """``081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-16...``"""
    parts = raw.rstrip("\n").split(" ", 5)
    if len(parts) < 6:
        return None
    date, time_, _pid, level, component, content = parts
    try:
        ts = datetime.strptime(date + time_, "%y%m%d%H%M%S")
    except ValueError:
        ts = None
    return HdfsLine(ts, level, component.rstrip(":"), content, BLOCK_RE.findall(content))


@dataclass
class Template:
    id: int
    tokens: list[str]
    count: int = 0

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


def _has_digits_or_mask(token: str) -> bool:
    return any(c.isdigit() for c in token) or token.startswith("<")


@dataclass
class TemplateMiner:
    """Fixed-depth (length -> first token) tree of templates with token-similarity merging."""

    sim_threshold: float = 0.5
    templates: list[Template] = field(default_factory=list)
    _tree: dict[tuple[int, str], list[int]] = field(default_factory=dict, repr=False)

    @staticmethod
    def _bucket(tokens: list[str]) -> tuple[int, str]:
        first = tokens[0] if tokens and not _has_digits_or_mask(tokens[0]) else WILDCARD
        return len(tokens), first

    @staticmethod
    def _similarity(template: list[str], tokens: list[str]) -> tuple[float, int]:
        same = sum(1 for a, b in zip(template, tokens, strict=True) if a == b and a != WILDCARD)
        wild = sum(1 for a in template if a == WILDCARD)
        return same / len(tokens), wild

    def match(self, key: str) -> int:
        """Template id for ``key`` without learning (UNKNOWN_ID if nothing is close enough)."""
        tokens = key.split()
        if not tokens:
            return UNKNOWN_ID
        best_id, best = UNKNOWN_ID, (-1.0, -1)
        for tid in self._tree.get(self._bucket(tokens), []):
            sim = self._similarity(self.templates[tid - 1].tokens, tokens)
            if sim[0] >= self.sim_threshold and (sim[0], -sim[1]) > (best[0], -best[1]):
                best_id, best = tid, sim
        return best_id

    def learn(self, key: str, weight: int = 1) -> int:
        tokens = key.split()
        if not tokens:
            return UNKNOWN_ID
        tid = self.match(key)
        if tid == UNKNOWN_ID:
            tid = len(self.templates) + 1
            self.templates.append(Template(tid, tokens, weight))
            self._tree.setdefault(self._bucket(tokens), []).append(tid)
            return tid
        tpl = self.templates[tid - 1]
        tpl.tokens = [a if a == b else WILDCARD for a, b in zip(tpl.tokens, tokens, strict=True)]
        tpl.count += weight
        return tid

    @property
    def n_features(self) -> int:
        return len(self.templates) + 1  # + UNKNOWN

    def feature_names(self) -> list[str]:
        return ["<UNKNOWN TEMPLATE>"] + [t.text for t in self.templates]

    def save(self, path: Path) -> None:
        payload = {
            "sim_threshold": self.sim_threshold,
            "templates": [
                {"id": t.id, "template": t.text, "count": t.count} for t in self.templates
            ],
        }
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TemplateMiner:
        data = json.loads(path.read_text(encoding="utf-8"))
        miner = cls(sim_threshold=data["sim_threshold"])
        for t in data["templates"]:
            tpl = Template(t["id"], t["template"].split(), t["count"])
            miner.templates.append(tpl)
            miner._tree.setdefault(cls._bucket(tpl.tokens), []).append(tpl.id)
        return miner
