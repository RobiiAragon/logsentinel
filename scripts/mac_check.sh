#!/usr/bin/env bash
# macOS end-to-end check: lint, tests, full HDFS pipeline (download 187 MB once, ~2 min),
# evaluation on CPU + CoreML, detection on the sample and a dashboard smoke test.
#   scripts/mac_check.sh            # full pipeline
#   QUICK=1 scripts/mac_check.sh    # skip download/training; use the committed models
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in python3.13 python3.12 python3.11; do command -v "$c" >/dev/null && PY="$c" && break; done
fi
[[ -n "$PY" ]] || { echo "Python 3.11+ not found (brew install python@3.12)"; exit 1; }
[[ -d .venv ]] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip --version >/dev/null 2>&1 || python -m ensurepip >/dev/null
python -m pip install -q --upgrade pip
python -m pip install -q -e ".[dev]"

echo "==> lint + tests"
ruff check . && ruff format --check . && pytest -q -W ignore::DeprecationWarning
logsentinel providers

if [[ -z "${QUICK:-}" ]]; then
  echo "==> prepare (Loghub HDFS_v1, 11.2M lines) + train + evaluate"
  logsentinel prepare
  logsentinel train
  logsentinel evaluate --providers cpu,coreml,qnn,cuda --out results/local-mac.json
fi

echo "==> detect on the committed sample"
rm -f /tmp/logsentinel-check.db
for m in autoencoder autoencoder-int8 autoencoder-w8a16 iforest; do
  logsentinel detect data/sample/hdfs_test_sample.log.gz --model "$m" \
    --labels data/sample/hdfs_test_sample_labels.csv --db /tmp/logsentinel-check.db | head -2
done

echo "==> dashboard smoke test"
logsentinel serve --db /tmp/logsentinel-check.db --port 8767 > /dev/null 2>&1 &
API=$!
trap 'kill $API 2>/dev/null || true' EXIT
for _ in $(seq 1 40); do curl -sf http://127.0.0.1:8767/api/runs >/dev/null && break; sleep 0.5; done
curl -sf "http://127.0.0.1:8767/api/alerts?limit=1" | python -c "import json,sys; a=json.load(sys.stdin); assert a; print('top alert:', a[0]['block'], a[0]['severity'])"
echo "OK - logsentinel serve --db /tmp/logsentinel-check.db"
