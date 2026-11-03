#!/bin/bash
# Resumable download of the FlyWire 783 Princeton synapse table (the proxy drops long single transfers).
U=https://storage.googleapis.com/flywire-data/codex/data/fafb/783/fafb_v783_princeton_synapse_table.csv.gz
F=dl/fafb_v783_princeton_synapse_table.csv.gz
for i in $(seq 1 60); do
  curl -sS -C - -o $F $U && break
  echo "retry $i at $(stat -c %s $F 2>/dev/null) bytes"; sleep 3
done
echo "final size $(stat -c %s $F) (expected 2695106039)"
