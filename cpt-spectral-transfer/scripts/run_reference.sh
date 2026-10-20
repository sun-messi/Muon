#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 CACHE_DIR NEW_OUTPUT_DIR {adam|muon|pion} {smoke|full}" >&2
  exit 2
fi
cache=$1
output=$2
method=$3
mode=$4
if [[ -e "$output" ]]; then
  echo "Refusing to overwrite existing output: $output" >&2
  exit 2
fi
case "$method" in
  adam) family=pythia_adam; schedule=none ;;
  muon) family=matched; schedule=0:muon5ns:0:0:0 ;;
  pion) family=matched; schedule=0:pion:3:2:.45 ;;
  *) echo "Unknown method: $method" >&2; exit 2 ;;
esac
extra=()
case "$mode" in
  smoke) stop=2; steps=0,2; save=0,1,2; every=1; extra+=(--smoke) ;;
  full) stop=610; steps=0,610; save=0,1,5,10,20,50,100,200,400,610; every=10 ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
python train_openwebmath.py --cache-dir "$cache" --output-dir "$output" \
  --seed 42 --rho-label rho20 --optimizer-family "$family" \
  --spectral-schedule "$schedule" --total-updates 610 --stop-after "$stop" \
  --matrix-lr 0.003 --aux-lr 0.001 --matrix-momentum 0.95 \
  --device-batch-size 8 --grad-accum-steps 4 --eval-every "$every" \
  --monitor-rows 32 --monitor-retention-every-validation \
  --full-validation-steps "$steps" --retention-steps "$steps" \
  --save-steps "$save" --diagnostic-updates 0 "${extra[@]}"
