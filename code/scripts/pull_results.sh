#!/bin/bash
# Copy the result files from the GPU instance to this machine (run from the repo root, e.g. in Git Bash).
#
#   bash scripts/pull_results.sh "ssh -p PORT -i path/to/key root@HOST"
#
# Pulls (all small, nothing is overwritten that is newer locally):
#   results/                          logs, tables, reports, split ids, qualitative files   (committed to the repo)
#   outputs/task1/eval_samples/       per-example GSM8K / benchmark outputs (answers, truncation flags)
#   outputs/task2/raw_predictions.json, outputs/task2/datasets.json
#   logs/                             queue status + per-step logs (copied to ./logs_remote/)
# Model weights / checkpoints / cached features are NOT pulled.
set -e
SSH_CMD="${1:?usage: bash scripts/pull_results.sh \"ssh -p PORT -i KEY root@HOST\"}"
REMOTE=/workspace/pa3
PATHS="results outputs/task1/eval_samples outputs/task2/raw_predictions.json outputs/task2/datasets.json"
mkdir -p logs_remote
echo "== results + per-example outputs =="
$SSH_CMD "cd $REMOTE && tar czf - \$(for p in $PATHS; do [ -e \$p ] && echo \$p; done)" | tar xzf -
echo "== queue status + logs =="
$SSH_CMD "cd /workspace/logs && tar czf - --exclude='*.pt' ." | tar xzf - -C logs_remote
echo "done. status:"; tail -5 logs_remote/queue_status.txt 2>/dev/null || true
du -sh results outputs/task1/eval_samples 2>/dev/null || true
