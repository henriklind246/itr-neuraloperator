#!/bin/bash
# Write held-out test records for each pub run against its locally generated,
# draw-family-disjoint dataset.
set -u
cd "$(dirname "$0")/.."
for B in forcing source source_itr interfaces; do
  SEED=$(basename "$(ls -d runs/pub_$B/config0/seed*)" | sed 's/seed//')
  echo "=== $B (seed $SEED) ==="
  .venv/bin/python -u scripts/write_test_records.py "runs/pub_$B/config0" \
      --seed "$SEED" --data-dir "data/pub_eval/$B" --eval-all-sims \
      --device "${PUB_DEVICE:-mps}"
  echo "exit=$?"
done
