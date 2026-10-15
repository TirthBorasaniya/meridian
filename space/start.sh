#!/bin/bash
# Start Qdrant, restore the latest accepted index snapshot, then serve the API and UI.
set -euo pipefail

DATA_DIR="${HOME}/data"
mkdir -p "${DATA_DIR}/qdrant" "${DATA_DIR}/snapshots"
cd /home/user/app

QDRANT__STORAGE__STORAGE_PATH="${DATA_DIR}/qdrant" \
QDRANT__STORAGE__SNAPSHOTS_PATH="${DATA_DIR}/snapshots" \
QDRANT__SERVICE__HOST=127.0.0.1 \
QDRANT__TELEMETRY_DISABLED=true \
  qdrant > "${DATA_DIR}/qdrant.log" 2>&1 &

for attempt in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:6333/readyz > /dev/null; then
    break
  fi
  sleep 1
done

started=$(date +%s)
meridian release restore --work-dir "${DATA_DIR}/release"
echo "{\"restore_seconds\": $(( $(date +%s) - started ))}" > "${DATA_DIR}/restore_timing.json"

# FORWARDED_ALLOW_IPS lists only private proxy ranges, so uvicorn takes the rightmost public
# address from X-Forwarded-For and a client cannot spoof its own
exec meridian serve --host 0.0.0.0 --port 7860
