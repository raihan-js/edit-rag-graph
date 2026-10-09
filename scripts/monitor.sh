#!/bin/bash
# Autonomous run monitor for EditBench-3Way milestone 2.
#   bash scripts/monitor.sh        # detached; logs data/logs/monitor.log
# Advances B -> C -> A on its own: waits for each results file, launches the
# next arm only when the GPU is free, writes results/STATUS.md at every
# transition and commits result files. Never edits code, never tunes.
cd "$(dirname "$0")/.."
LOG=data/logs/monitor.log
mkdir -p data/logs
log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }
gpu_free() { ! pgrep -f "run_arm[ABC].py" >/dev/null; }
status() { printf "# Run status (%s)\n\n- armB: %s\n- armC: %s\n- armA: %s\n" \
  "$(date '+%Y-%m-%d %H:%M')" "$1" "$2" "$3" > results/STATUS.md; }

armB=pending; armC=pending; armA=pending; hold=pending
[ -f results/armB.json ] && armB=done
status "$armB" "$armC" "$armA"
log "monitor started (armB=$armB)"

while true; do
  if [ "$armB" != done ] && [ -f results/armB.json ] && gpu_free; then
    armB=done; status "$armB" "$armC" "$armA"
    log "armB finished -> launching armC"
    setsid nohup .venv/bin/python scripts/run_armC.py > /tmp/armC.log 2>&1 < /dev/null &
    armC=running; status "$armB" "$armC" "$armA"
  fi
  if [ "$armC" = running ] && [ -f results/armC.json ] && gpu_free; then
    armC=done; status "$armB" "$armC" "$armA"
    log "armC finished -> launching armA (ROME/MEMIT, slow)"
    setsid nohup .venv/bin/python scripts/run_armA.py > /tmp/armA.log 2>&1 < /dev/null &
    armA=running; status "$armB" "$armC" "$armA"
  fi
  if [ "$armA" = running ] && grep -q '"memit": {' results/armA.json 2>/dev/null && gpu_free; then
    armA=done; status "$armB" "$armC" "$armA"
    log "armA finished -> hold-probe patch (base model only)"
    setsid nohup .venv/bin/python scripts/rehold.py > /tmp/rehold.log 2>&1 < /dev/null &
    hold=running; status "$armB $armC $armA hold:$hold"
  fi
  if [ "$hold" = running ] && grep -q hold_probe_fix results/armB.json 2>/dev/null; then
    hold=done
    log "hold patch finished -> committing everything"
    git add -A && git commit -qm "Arm results + hold-probe patch (autonomous monitor)" || true
    log "ALL DONE"
    exit 0
  fi
  sleep 120
done
