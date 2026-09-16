#!/bin/bash
# Publication analysis from held-out test records alone: the source-time sweep
# check, the global-field figures, and the primary results table.
#
# A cluster has the records but no manifest.yaml, so every step here is driven
# by --runs-root discovery instead. Point it at the directory holding
# <experiment>/seed*/test_records.csv; it defaults to runs/.
#
#   scripts/_pub_analysis.sh
#   scripts/_pub_analysis.sh /scratch/$USER/runs
#   scripts/_pub_analysis.sh /scratch/$USER/runs --run forcing=/scratch/$USER/runs/pub_forcing/config0
#   PUB_PYTHON=python3 scripts/_pub_analysis.sh /scratch/$USER/runs
#
# Arguments after the root are forwarded to every step, which is how you
# disambiguate a benchmark that has more than one evaluated experiment.
#
# Steps report their exit code rather than aborting the run: the sweep returns 2
# when the plotted source time is atypical and the figure and table steps return
# 2 when they render degraded. Those are findings, and they are worth seeing
# together in one log.
set -u
cd "$(dirname "$0")/.."

ROOT="${1:-runs}"
[ $# -gt 0 ] && shift
OUT="${PUB_OUT:-figures/pub}"

# A cluster usually has no .venv; there the interpreter is whatever the loaded
# module or conda environment provides. Resolve it once and say so, rather than
# letting all three steps fail with a bare 127.
if [ -n "${PUB_PYTHON:-}" ]; then
  PY="$PUB_PYTHON"
elif [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "No interpreter: set PUB_PYTHON=/path/to/python" >&2
  exit 127
fi
if ! "$PY" -c 'import pandas, matplotlib' 2>/dev/null; then
  echo "$PY cannot import pandas and matplotlib; set PUB_PYTHON to an environment that can" >&2
  exit 127
fi
echo "python: $("$PY" -c 'import sys; print(sys.executable)')"
echo "runs:   $ROOT"
echo

echo "=== source-time sweep: is F32's plotted t_s* representative? ==="
"$PY" -u scripts/inspect_source_time_sweep.py --runs-root "$ROOT" "$@"
echo "exit=$?"

echo
echo "=== global-field figures: F27, F28, F32 ==="
"$PY" -u -m visual.pub --global-field --runs-root "$ROOT" "$@"
echo "exit=$?"

echo
echo "=== T01 primary results ==="
"$PY" -u -m visual.pub --runs-root "$ROOT" --table T01_primary_results --out "$OUT" "$@"
STATUS=$?
if [ "$STATUS" -eq 1 ]; then
  # Usually the seed cohort: the strict table wants three, and one seed per
  # benchmark is the normal state of a single sweep. The descriptive table is
  # the honest rendering of that -- same numbers, no inferential claim attached.
  # Any other unmet requirement is named on stderr just above this line, and
  # the fallback will fail on it too rather than paper over it.
  echo "strict T01 unmet (see above); retrying the descriptive table"
  "$PY" -u -m visual.pub --runs-root "$ROOT" \
      --table T01_primary_results_descriptive --out "$OUT" "$@"
  STATUS=$?
fi
echo "exit=$STATUS"
