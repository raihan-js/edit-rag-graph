#!/bin/bash
# Autonomous run monitor for EditBench-3Way milestone 2.
#   bash scripts/monitor.sh        # detached; logs data/logs/monitor.log
# Order: MEMIT (fresh process) -> hold patch -> commit. ROME already scored.
# Never edits code, never tunes. Status in results/STATUS.md.
cd "$(dirname "$0")/.."
mkdir -p data/logs
LOG=data/logs/monitor.log
log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }
status() {
  printf '# Run status (%s)\n\n- armB: done\n- armC: done\n- rome: done\n- memit: %s\n- hold_patch: %s\n' \
    "$(date '+%Y-%m-%d %H:%M')" "$1" "$2" > results/STATUS.md
}
running() { pgrep -f "$1" >/dev/null 2>&1; }

memit=pending
hold=pending
if grep -q '"memit": {' results/armA.json 2>/dev/null; then memit=done; fi
if grep -q hold_probe_fix results/armB.json 2>/dev/null; then hold=done; fi
status "$memit" "$hold"
log "monitor started (memit=$memit hold=$hold)"

while true; do
  if [ "$memit" = pending ] && ! running "run_armA_memit" && ! running "run_armA\.py"; then
    log "launching MEMIT (fresh process)"
    setsid nohup .venv/bin/python scripts/run_armA_memit.py > /tmp/armA_memit.log 2>&1 < /dev/null &
    memit=running
    status "$memit" "$hold"
  fi
  if [ "$memit" = running ] && grep -q '"memit": {' results/armA.json 2>/dev/null \
     && ! running "run_armA_memit"; then
    memit=done
    status "$memit" "$hold"
    log "MEMIT finished"
  fi
  if [ "$memit" = done ] && [ "$hold" = pending ] && ! running "rehold"; then
    log "launching hold-probe patch"
    setsid nohup .venv/bin/python scripts/rehold.py > /tmp/rehold.log 2>&1 < /dev/null &
    hold=running
    status "$memit" "$hold"
  fi
  if [ "$hold" = running ] && grep -q hold_probe_fix results/armB.json 2>/dev/null \
     && ! running "rehold"; then
    hold=done
    status "$memit" "$hold"
    log "hold patch finished -> committing everything"
    git add -A && git commit -qm "Arm results + hold-probe patch (autonomous monitor)" || true
    log "ALL DONE"
    exit 0
  fi
  sleep 120
done
