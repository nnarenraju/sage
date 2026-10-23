#!/bin/bash
# Memory watchdog for the ASD jobs. Logs headroom; kills our make_asds_single
# processes if available RAM stays critically low (imminent OOM, swap~0).
FLOOR_MB=1000      # kill threshold for MemAvailable
STRIKES_MAX=3      # consecutive low readings (~60s) before acting
INTERVAL=20
strikes=0
log=/home/nnarenraju/Research/sage/runs/asd_memguard.log
echo "$(date +%T) memguard started (floor=${FLOOR_MB}MB, ${STRIKES_MAX} strikes)" >> "$log"
while true; do
  pids=$(pgrep -u "$USER" -f make_asds_single)
  if [ -z "$pids" ]; then
    echo "$(date +%T) all ASD jobs finished/gone; memguard exiting" >> "$log"; break
  fi
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  ourrss=$(ps -u "$USER" -o rss,cmd | grep make_asds_single | grep -v grep | awk '{s+=$1} END{printf "%d", s/1024}')
  njobs=$(echo "$pids" | wc -w)
  echo "$(date +%T) avail=${avail}MB our_jobs=${njobs} our_rss=${ourrss}MB strikes=${strikes}" >> "$log"
  if [ "$avail" -lt "$FLOOR_MB" ]; then
    strikes=$((strikes+1))
    if [ "$strikes" -ge "$STRIKES_MAX" ]; then
      echo "$(date +%T) !!! avail<${FLOOR_MB}MB sustained -> KILLING ASD jobs to avoid OOM" >> "$log"
      pkill -u "$USER" -f make_asds_single
      break
    fi
  else
    strikes=0
  fi
  sleep "$INTERVAL"
done
