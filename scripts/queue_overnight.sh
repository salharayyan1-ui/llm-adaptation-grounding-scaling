#!/bin/bash
# Unattended end-to-end queue for ATML PA3 on ONE 24 GB GPU (RTX 3090 / 4090) with >= ~60 GB disk.
#
#  * idempotent: every step has a "done" marker and is skipped when it exists
#  * training steps resume from the latest checkpoint
#  * a failing step is retried once, then the queue moves on (one failure never blocks the rest)
#  * Full FT / L2-SP use the SPEC checkpoints (updates 312/625/937/1250 incl. optimizer state, ~7.2 GB each) while
#    training; after their evaluations the optimizer states and unused checkpoints are pruned (peak disk ~45 GB).
#    If the disk is too small they fall back to lean checkpoints (documented in the status log).
#
# Start / restart:  tmux new -d -s night "bash scripts/queue_overnight.sh >> /workspace/logs/overnight.out 2>&1"
# Status:           /workspace/logs/queue_status.txt        Per-step logs: /workspace/logs/<step>.log

source /venv/main/bin/activate
cd "${ROOT:-/workspace/pa3}" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false
LOG=/workspace/logs
mkdir -p "$LOG" results/task1/eval
ST="$LOG/queue_status.txt"
T1CFG=configs/task1_config.yaml

say() { echo "[$(date -u '+%F %T')] $*" | tee -a "$ST"; }
free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }

step() {  # step NAME MARKER cmd...
  local name=$1 marker=$2 a rc
  shift 2
  if [ -n "$marker" ] && [ -e "$marker" ]; then say "SKIP  $name (done)"; return 0; fi
  for a in 1 2; do
    say "START $name (attempt $a, free disk $(free_gb)G)"
    "$@" >> "$LOG/$name.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && { [ -z "$marker" ] || [ -e "$marker" ]; }; then say "DONE  $name"; return 0; fi
    say "FAIL  $name rc=$rc (attempt $a)"
    sleep 20
  done
  return 1
}

latest() {  # latest "<glob>" -> path with the highest trailing step number (empty if none)
  local d n best="" bn=-1
  for d in $1; do
    [ -e "$d" ] || continue
    n=${d##*_}; n=${n%.pt}
    if [ "$n" -gt "$bn" ] 2>/dev/null; then bn=$n; best=$d; fi
  done
  echo "$best"
}

t1_train() {  # t1_train VARIANT CONFIG [fresh]
  local v=$1 cfg=$2 mode=$3 ck=""
  [ "$mode" = "fresh" ] || ck=$(latest "outputs/task1/$v/checkpoint_step_*")
  if [ -n "$ck" ]; then
    say "resuming $v from $ck"
    python scripts/run_task1.py --config "$cfg" --action train --variant "$v" --resume_from "$ck" --gpu 0
  else
    python scripts/run_task1.py --config "$cfg" --action train --variant "$v" --gpu 0
  fi
}

t2_train() {  # t2_train ACTION OUTDIR   (ACTION = train | compress)
  local ck; ck=$(latest "$2/checkpoint_step_*.pt")
  if [ -n "$ck" ]; then
    say "resuming Task 2 $1 from $ck"
    python scripts/run_task2.py --action "$1" --resume_from "$ck" --gpu 0
  else
    python scripts/run_task2.py --action "$1" --gpu 0
  fi
}

free_disk_for() {  # free_disk_for NEED_GB -> deletes regenerable files, cheapest first, until enough space
  local need=$1
  [ "$(free_gb)" -ge "$need" ] && return 0
  rm -rf data/_downloads;                                              [ "$(free_gb)" -ge "$need" ] && return 0
  for v in lora_16 lora_4 oplora_16; do                                # intermediate recovery checkpoints of finished adapter runs
    if [ -e "results/task1/$v/summary.json" ]; then rm -rf "outputs/task1/$v/checkpoint_step_312" "outputs/task1/$v/checkpoint_step_937"; fi
  done;                                                                [ "$(free_gb)" -ge "$need" ] && return 0
  if [ -e results/task2/caching_stats.json ]; then                     # only once the SigLIP features exist
    rm -rf data/coco2014/train2014;                                    [ "$(free_gb)" -ge "$need" ] && return 0
    rm -rf data/coco2014/annotations data/vqav2;                       [ "$(free_gb)" -ge "$need" ] && return 0
  fi
  return 1
}

prune_full() {  # prune_full VARIANT KEEP_STEP...  -> drop optimizer states + every other checkpoint (call after the evals)
  local v=$1 d keep k drop; shift
  d="outputs/task1/$v"
  [ -e "results/task1/$v/summary.json" ] || return 0
  for dir in "$d"/checkpoint_step_*; do
    [ -e "$dir" ] || continue
    drop=1
    for k in "$@"; do [ "${dir##*_}" = "$k" ] && drop=0; done
    if [ $drop -eq 1 ]; then rm -rf "$dir"; else rm -f "$dir/optimizer_state.pt"; fi
  done
  say "pruned $v checkpoints (kept weights of: $*) -> free disk $(free_gb)G"
}

run_full() {  # run_full VARIANT LEAN_SAVE_UPDATES   (e.g. full_ft "625, 1250")
  local v=$1 lean=$2 cfg=$T1CFG mode=""
  [ -e "results/task1/$v/summary.json" ] && return 0
  if ! free_disk_for 32; then
    say "WARN $v: only $(free_gb)G free (spec checkpoints need ~32G): LEAN checkpoints [$lean], no optimizer state"
    cfg=/workspace/cfg_lean_$v.yaml; mode=fresh
    sed -e "s/save_updates: \[.*\]/save_updates: [$lean]/" -e 's/save_optimizer_state: true/save_optimizer_state: false/' "$T1CFG" > "$cfg"
    free_disk_for 8 || { say "SKIP train_$v: not enough disk even for lean checkpoints"; return 1; }
  fi
  step "train_$v" "results/task1/$v/summary.json" t1_train "$v" "$cfg" $mode
}

ev() { step "eval_$1" "results/task1/eval/$1.json" python scripts/run_task1.py --action evaluate --conditions "$1" --gpu 0; }

say "=============== queue start (free disk $(free_gb)G) ==============="

# ---------------- Task 1 ----------------
ev baseline
for v in lora_16 lora_4 oplora_16; do
  step "train_$v" "results/task1/$v/summary.json" t1_train "$v" "$T1CFG"
  ev "${v}_final"
done
run_full full_ft "625, 1250"
ev full_ft_midpoint; ev full_ft_final
prune_full full_ft 625 1250
run_full full_ft_l2_sp "1250"
ev full_ft_l2_sp_final
prune_full full_ft_l2_sp 1250

# ---------------- Task 2 (images were started in tmux 'imgs' by the bootstrap; this step is idempotent) ----------------
while tmux has-session -t imgs 2>/dev/null; do sleep 30; done
step t2_images "" python scripts/download_assets.py --what images
step t2_cache results/task2/caching_stats.json python scripts/run_task2.py --action cache_features --gpu 0
step t2_train results/task2/training_summary.json t2_train train outputs/task2
step t2_eval results/task2/final_evaluation_report.json python scripts/run_task2.py --action evaluate --gpu 0

# ---------------- Task 1 analyses over every condition that exists ----------------
avail=""
for c in baseline full_ft_midpoint full_ft_final full_ft_l2_sp_final lora_4_final lora_16_final oplora_16_final; do
  [ -e "results/task1/eval/$c.json" ] && avail="$avail,$c"
done
avail=${avail#,}
nobase=$(echo "$avail" | sed -e 's/^baseline,//' -e 's/^baseline$//')
say "conditions available: $avail"
step diag_all "" python scripts/run_task1.py --action diagnostics --conditions "$nobase" --gpu 0
step qual_all "" python scripts/run_task1.py --action qualitative --conditions "$avail" --gpu 0
step aggregate "" python scripts/run_task1.py --action aggregate

# ---------------- Task 2: optional 49-token run last ----------------
step t2_compress results/task2_49tok/final_evaluation_report.json t2_train compress outputs/task2_49tok

say "=============== queue finished (free disk $(free_gb)G) ==============="
{
  echo "== markers at $(date -u '+%F %T') =="
  for f in results/task1/*/summary.json results/task1/eval/*.json results/task2/caching_stats.json \
           results/task2/training_summary.json results/task2/final_evaluation_report.json \
           results/task2_49tok/final_evaluation_report.json; do
    [ -e "$f" ] && echo "OK      $f" || echo "MISSING $f"
  done
} >> "$ST"
