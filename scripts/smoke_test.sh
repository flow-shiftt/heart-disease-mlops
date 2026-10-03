#!/usr/bin/env bash
# End-to-end check of a running API: health, readiness, one prediction, bad input, metrics.
# Usage: scripts/smoke_test.sh [base_url]     (default http://localhost:8000)
set -euo pipefail
BASE="${1:-http://localhost:8000}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== waiting for $BASE/ready"
for i in $(seq 1 30); do
  if curl -fs "$BASE/ready" >/dev/null; then break; fi
  [ "$i" -eq 30 ] && { echo "API never became ready"; exit 1; }
  sleep 2
done

echo "== GET /health";   curl -fsS "$BASE/health"; echo
echo "== POST /predict (high-risk sample)"
curl -fsS -X POST "$BASE/predict" -H 'content-type: application/json' \
  -d @"$HERE/../sample_request.json" | tee /tmp/cardiorisk_pred.json; echo
grep -q '"confidence"' /tmp/cardiorisk_pred.json
grep -q '"prediction"' /tmp/cardiorisk_pred.json

echo "== POST /predict (invalid payload must be rejected with 422)"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" \
  -H 'content-type: application/json' -d '{"age": 500}')
[ "$code" = "422" ] || { echo "expected 422, got $code"; exit 1; }
echo "422 OK"

echo "== GET /metrics (excerpt)"
curl -fsS "$BASE/metrics" | grep -E '^cardiorisk_(predictions_total|model_info)' | head
echo "SMOKE TEST PASSED"
