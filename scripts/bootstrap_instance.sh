#!/bin/bash
# One-shot setup + launch for a FRESH Vast.ai PyTorch instance (24 GB GPU, >= ~60 GB disk).
# Prerequisite: the repo is at /workspace/pa3 (e.g. `tar czf - --exclude=.git --exclude=outputs --exclude=data . | ssh ... "mkdir -p /workspace/pa3 && tar xzf - -C /workspace/pa3"`).
# Idempotent: safe to re-run after an interruption (it never restarts a running tmux session).
#
#   bash /workspace/pa3/scripts/bootstrap_instance.sh            # deps, smoke test, data prep, image download, queue
#   SKIP_SMOKE=1 bash scripts/bootstrap_instance.sh               # skip the ~3 min GPU smoke test
set -u
ROOT=/workspace/pa3
cd "$ROOT" || { echo "repo not found at $ROOT"; exit 1; }
source /venv/main/bin/activate
mkdir -p /workspace/logs
export HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false

echo "== disk / gpu =="
df -h / | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
size_gb=$(df -BG --output=size / | tail -1 | tr -dc '0-9')
if [ "$size_gb" -lt 50 ]; then
  echo "WARNING: only ${size_gb}G of disk. The queue falls back to lean Full FT checkpoints (see scripts/queue_overnight.sh)."
fi

echo "== dependencies =="
if ! python -c "import lm_eval, peft, math_verify, transformers" 2>/dev/null; then
  uv pip install -r requirements.txt 2>&1 | tail -3
fi
python -c "
import torch, transformers, peft, lm_eval, datasets
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__,
      '| peft', peft.__version__, '| lm_eval', lm_eval.__version__, '| datasets', datasets.__version__)
assert torch.cuda.is_available()
assert transformers.__version__.startswith('4.'), 'code is written against transformers 4.x'
"
[ $? -eq 0 ] || { echo "environment check failed"; exit 1; }

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "== GPU smoke test (tiny random models, ~3 min) =="
  python scripts/smoke_test_gpu.py > /workspace/logs/smoke.log 2>&1
  tail -1 /workspace/logs/smoke.log
  grep -q -E "^([0-9]+)/\1 checks passed" /workspace/logs/smoke.log || { echo "smoke test FAILED - see /workspace/logs/smoke.log"; grep "^\[FAIL\]" /workspace/logs/smoke.log; exit 1; }
fi

echo "== Task 2 annotations + pairing =="
[ -e data/vqav2/v2_mscoco_val2014_complementary_pairs.json ] && [ -e data/coco2014/annotations/instances_val2014.json ] \
  || python scripts/download_assets.py --what annotations,vqa
[ -e outputs/task2/datasets.json ] || python scripts/run_task2.py --action prepare > /workspace/logs/t2_prep.log 2>&1
ls -la outputs/task2/datasets.json results/task2/split_ids.json

echo "== COCO images (background, throttled host: ~3 h) =="
tmux has-session -t imgs 2>/dev/null || tmux new-session -d -s imgs \
  "bash -c 'source /venv/main/bin/activate; cd $ROOT; python scripts/download_assets.py --what images' >> /workspace/logs/t2_images.log 2>&1"

echo "== queue =="
tmux has-session -t night 2>/dev/null && echo "queue already running (tmux: night)" || {
  tmux new-session -d -s night "bash $ROOT/scripts/queue_overnight.sh >> /workspace/logs/overnight.out 2>&1"
  echo "queue started (tmux: night)"
}
sleep 3
tmux ls
echo "status file: /workspace/logs/queue_status.txt"
cat /workspace/logs/queue_status.txt 2>/dev/null | tail -5
