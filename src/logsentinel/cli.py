"""``logsentinel`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from . import __version__

PROCESSED = Path("data/processed")
MODELS = Path("models")
SAMPLE = Path("data/sample")


def cmd_prepare(a: argparse.Namespace) -> int:
    from .pipeline import prepare

    prepare(a.processed, a.models_dir, a.sample_dir, a.sample_blocks, a.limit)
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    from .pipeline import train

    report = train(a.processed, a.models_dir, epochs=a.epochs)
    print(f"ONNX vs NumPy max |diff| = {report['onnx_max_abs_diff']:.2e}")
    return 0


def cmd_evaluate(a: argparse.Namespace) -> int:
    from . import device_info, providers
    from .pipeline import evaluate

    usable, skipped = providers.resolve(a.providers)
    for alias, why in skipped.items():
        print(f"[skip] {alias}: {why}")
    results = evaluate(a.processed, a.models_dir, usable)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(
            json.dumps(
                {
                    "created": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
                    "version": __version__,
                    "device": device_info.collect(),
                    "skipped_providers": skipped,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {a.out}")
    return 0


def cmd_detect(a: argparse.Namespace) -> int:
    from . import providers
    from .detect import detect

    usable, skipped = providers.resolve(a.provider)
    if not usable:
        print(
            f"provider {a.provider!r} unavailable: {skipped.get(a.provider, 'unknown')}",
            file=sys.stderr,
        )
        return 2
    run = detect(a.log, a.models_dir, a.model, a.db, a.provider, a.labels, threshold=a.threshold)
    print(
        f"run #{run['id']}: {run['n_lines']:,} lines, {run['n_sessions']:,} sessions, "
        f"{run['n_alerts']} alerts in {run['seconds']:.2f}s ({run['lines_per_s']:,.0f} lines/s)"
    )
    if run["metrics"]:
        m = run["metrics"]
        print(
            f"  vs labels: precision {m['precision']:.3f} recall {m['recall']:.3f} "
            f"F1 {m['f1']:.3f} (tp={m['tp']} fp={m['fp']} fn={m['fn']})"
        )
    print(f"  alerts stored in {a.db}; view them with `logsentinel serve --db {a.db}`")
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    from . import export, store

    if not a.db.exists():
        print(f"{a.db} not found; run `logsentinel detect` first", file=sys.stderr)
        return 2
    con = store.connect(a.db)
    if a.run is None:
        all_runs = store.runs(con)
        run = all_runs[0] if all_runs else None
    else:
        run = store.run(con, a.run)
    if run is None:
        print("no such detection run", file=sys.stderr)
        return 2
    alerts = store.alerts(con, run["id"], a.severity, limit=a.limit)
    lines = export.ndjson(run, alerts) if a.format == "ecs" else export.csv_rows(run, alerts)
    if a.out:
        with a.out.open("w", encoding="utf-8") as fh:
            fh.writelines(lines)
        print(f"wrote {len(alerts)} alerts from run #{run['id']} to {a.out}", file=sys.stderr)
    else:
        sys.stdout.writelines(lines)
    return 0


def cmd_card(a: argparse.Namespace) -> int:
    from .card import write

    print(f"wrote {write(a.models_dir, a.results, a.out)}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    print(f"LogSentinel dashboard on http://{a.host}:{a.port}")
    uvicorn.run(create_app(a.db), host=a.host, port=a.port, log_level="warning")
    return 0


def cmd_providers(_: argparse.Namespace) -> int:
    from . import providers

    rt = providers.ort()
    print(f"onnxruntime {rt.__version__}: {', '.join(rt.get_available_providers())}")
    usable, skipped = providers.resolve(",".join(providers.EP_ALIASES))
    for alias in usable:
        print(f"  [ok]   {alias}")
    for alias, why in skipped.items():
        print(f"  [skip] {alias}: {why}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="logsentinel", description="Log anomaly detection for SOC")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("prepare", help="download + parse Loghub HDFS_v1 into features")
    pr.add_argument("--processed", type=Path, default=PROCESSED)
    pr.add_argument("--models-dir", type=Path, default=MODELS)
    pr.add_argument("--sample-dir", type=Path, default=SAMPLE)
    pr.add_argument("--sample-blocks", type=int, default=1500)
    pr.add_argument("--limit", type=int, help="only read the first N lines (quick runs)")
    pr.set_defaults(func=cmd_prepare)

    t = sub.add_parser("train", help="train Isolation Forest + autoencoder, export ONNX + INT8")
    t.add_argument("--processed", type=Path, default=PROCESSED)
    t.add_argument("--models-dir", type=Path, default=MODELS)
    t.add_argument("--epochs", type=int, default=8)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="test-split metrics and speed for every model x EP")
    e.add_argument("--processed", type=Path, default=PROCESSED)
    e.add_argument("--models-dir", type=Path, default=MODELS)
    e.add_argument("--providers", default="cpu,coreml,qnn,cuda,nnapi,xnnpack")
    e.add_argument("--out", type=Path)
    e.set_defaults(func=cmd_evaluate)

    d = sub.add_parser("detect", help="score a log file and store alerts in SQLite")
    d.add_argument("log", type=Path, help="HDFS-format log (.log or .log.gz)")
    d.add_argument(
        "--model",
        default="autoencoder",
        choices=["autoencoder", "autoencoder-int8", "autoencoder-w8a16", "iforest"],
    )
    d.add_argument("--models-dir", type=Path, default=MODELS)
    d.add_argument("--provider", default="cpu")
    d.add_argument("--labels", type=Path, help="optional BlockId,Label CSV to score the run")
    d.add_argument("--db", type=Path, default=Path("alerts.db"))
    d.add_argument(
        "--threshold",
        type=float,
        help="override the validation-chosen alert threshold (raise it to cut alert volume)",
    )
    d.set_defaults(func=cmd_detect)

    x = sub.add_parser("export", help="export a run's alerts as ECS NDJSON or CSV (for a SIEM)")
    x.add_argument("--db", type=Path, default=Path("alerts.db"))
    x.add_argument("--run", type=int, help="run id (default: latest)")
    x.add_argument("--format", choices=["ecs", "csv"], default="ecs")
    x.add_argument("--severity", choices=["critical", "high", "medium"])
    x.add_argument("--limit", type=int, default=100_000)
    x.add_argument("-o", "--out", type=Path, help="file to write (default: stdout)")
    x.set_defaults(func=cmd_export)

    c = sub.add_parser("card", help="regenerate MODEL_CARD.md from models/ and a results file")
    c.add_argument("--models-dir", type=Path, default=MODELS)
    c.add_argument("--results", type=Path, default=Path("results/mac-m5max.json"))
    c.add_argument("-o", "--out", type=Path, default=Path("MODEL_CARD.md"))
    c.set_defaults(func=cmd_card)

    s = sub.add_parser("serve", help="alert dashboard (FastAPI)")
    s.add_argument("--db", type=Path, default=Path("alerts.db"))
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    sub.add_parser("providers", help="list ONNX Runtime execution providers").set_defaults(
        func=cmd_providers
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
