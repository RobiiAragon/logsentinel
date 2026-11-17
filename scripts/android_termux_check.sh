#!/data/data/com.termux/files/usr/bin/bash
# LogSentinel on Android via Termux (target: Galaxy Tab S9 Ultra, Snapdragon 8 Gen 2).
# Inference only: NumPy + ONNX Runtime (CPU / XNNPACK / NNAPI). No scikit-learn needed.
#
#   pkg install -y git && git clone https://github.com/RobiiAragon/logsentinel
#   cd logsentinel && bash scripts/android_termux_check.sh          # sample, ~1 min
#   FULL=1 bash scripts/android_termux_check.sh                      # full HDFS test split
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "${PREFIX:-}" == *com.termux* ]] || { echo "This script is meant for Termux."; exit 1; }

pkg update -y >/dev/null
pkg install -y python python-numpy python-onnxruntime git >/dev/null
python -m pip install -q --no-deps -e .
logsentinel providers

TAG=$(getprop ro.product.model | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
DB="results/android-$TAG.db"
mkdir -p results && rm -f "$DB"
echo "==> detection on the committed HDFS sample (parse + score, CPU)"
for m in autoencoder autoencoder-int8 autoencoder-w8a16 iforest; do
  logsentinel detect data/sample/hdfs_test_sample.log.gz --model "$m" \
    --labels data/sample/hdfs_test_sample_labels.csv --db "$DB" | head -2
done
for p in xnnpack nnapi; do
  echo "--> autoencoder on $p"
  logsentinel detect data/sample/hdfs_test_sample.log.gz --model autoencoder --provider "$p" \
    --labels data/sample/hdfs_test_sample_labels.csv --db "$DB" | head -2 || echo "   ($p unavailable)"
done

if [[ -n "${FULL:-}" ]]; then
  echo "==> full test split: download 187 MB + parse 11.2M lines on the tablet"
  logsentinel prepare --models-dir data/processed
  cmp -s data/processed/templates.json models/templates.json \
    && echo "templates identical to the committed ones" \
    || { echo "templates differ from models/templates.json; aborting"; exit 1; }
  logsentinel evaluate --providers cpu,xnnpack,nnapi --out "results/android-$TAG.json"
fi
echo "done. Results: $DB ${FULL:+results/android-$TAG.json}"
