#!/usr/bin/env bash
# Evaluate an agent config (submission dir) on the Gemma 4 Developer Agent tasks with the official harness.
#
# Endpoint mode: any OpenAI-compatible model endpoint, run in parallel workers.
#   ./evaluate.sh --agent agents/shell_v2 --endpoint https://host/v1 --model NAME [--api-key KEY]
#                 [--backend passthrough|strict|openrouter] [--extra-body JSON] [--sandbox subprocess|docker]
#                 [--workers 16] [--reps 1] [--tasks-file dev_tasks.jsonl] [--tasks ID ...] [--name NAME]
#   e.g. OpenRouter: --endpoint https://openrouter.ai/api/v1 --backend openrouter --model google/gemma-4-31b-it
#   The workers run on an internal docker network with no internet, as on Kaggle. They reach the model only
#   through a gateway container, which holds the API key and logs every request/response under runs/NAME/model_logs.
#
# Local mode (no --endpoint): serves the agent's declared model with vLLM on --gpus, emulating 4x L4 (Kaggle-like).
#   ./evaluate.sh --agent agents/shell_v2 --gpus 0,1,2,3 [--tasks-file ...]
#
# The API key can also come from the EVAL_API_KEY environment variable, which keeps it off the command line.
# Results: runs/NAME/<worker>/{results.jsonl,submission.parquet,results/traces}, runs/NAME/summary.txt (pass@k).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)

AGENT= ENDPOINT= MODEL= BACKEND=passthrough EXTRA_BODY= SANDBOX=subprocess WORKERS=16 REPS=1 NAME= GPUS=0,1,2,3
TASKS_FILE="$ROOT/dev_tasks.jsonl" TASK_IDS=() FIRST=
IMAGE=${IMAGE:-gemma4-kaggle:v170-plus}
SANDBOX_IMAGE=${SANDBOX_IMAGE:-swebench-sandbox:latest}
DATA_DIR=${DATA_DIR:-$ROOT/data}
MODELS_DIR=${MODELS_DIR:-$ROOT/models}
API_KEY=${EVAL_API_KEY:-}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent) AGENT=$2; shift 2;;
    --endpoint) ENDPOINT=$2; shift 2;;
    --model) MODEL=$2; shift 2;;
    --api-key) API_KEY=$2; shift 2;;
    --backend) BACKEND=$2; shift 2;;
    --extra-body) EXTRA_BODY=$2; shift 2;;
    --sandbox) SANDBOX=$2; shift 2;;
    --workers) WORKERS=$2; shift 2;;
    --reps) REPS=$2; shift 2;;
    --name) NAME=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;;
    --tasks-file) TASKS_FILE=$2; shift 2;;
    --first) FIRST=$2; shift 2;;
    --tasks) shift; while [[ $# -gt 0 && "$1" != --* ]]; do TASK_IDS+=("$1"); shift; done;;
    -h|--help) sed -n '2,16p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n "$AGENT" ]] || { echo "--agent is required" >&2; exit 2; }
[[ "$SANDBOX" == subprocess || "$SANDBOX" == docker ]] || { echo "--sandbox must be subprocess or docker" >&2; exit 2; }
AGENT=$(realpath "$AGENT"); TASKS_FILE=$(realpath "$TASKS_FILE")
[[ -f "$AGENT/agent.yaml" ]] || { echo "no agent.yaml in $AGENT" >&2; exit 2; }
[[ -f "$DATA_DIR/tasks.jsonl" ]] || { echo "competition data not found in $DATA_DIR (run scripts/setup.sh)" >&2; exit 2; }
NAME=${NAME:-$(basename "$AGENT")_$(date +%m%d_%H%M%S)}
OUT="$ROOT/runs/$NAME"; mkdir -p "$OUT"
cp -r "$AGENT" "$OUT/agent_config"

EXTRA=(--tasks-file /kaggle/tasks.jsonl --sandbox "$SANDBOX" --sandbox-image "$SANDBOX_IMAGE")
[[ -n "$FIRST" ]] && EXTRA+=(--first "$FIRST")
[[ ${#TASK_IDS[@]} -gt 0 ]] && EXTRA+=(--tasks "${TASK_IDS[@]}")
MOUNTS=(-v "$DATA_DIR:/kaggle/input/competitions/gemma-4-developer-agent:ro" -v "$AGENT:/kaggle/agent:ro"
        -v "$TASKS_FILE:/kaggle/tasks.jsonl:ro" -v "$ROOT/eval/run_eval.py:/opt/run_eval.py:ro")
if [[ "$SANDBOX" == docker ]]; then
  docker image inspect "$SANDBOX_IMAGE" >/dev/null 2>&1 || { echo "sandbox image $SANDBOX_IMAGE missing (scripts/setup.sh --sandbox-image)" >&2; exit 2; }
  MOUNTS+=(-v /var/run/docker.sock:/var/run/docker.sock)
fi
COMMON=(--ipc host --shm-size 16g --ulimit memlock=-1)

fix_owner() { docker run --rm -v "$OUT:/o" --entrypoint chown "$IMAGE" -R "$(id -u):$(id -g)" /o >/dev/null 2>&1 || true; }

# ---------------- local vLLM mode ----------------
if [[ -z "$ENDPOINT" ]]; then
  for m in "$MODELS_DIR"/gemma-4-*; do MOUNTS+=(-v "$m:/kaggle/input/models/google/gemma-4/other/$(basename "$m")/2:ro"); done
  for r in $(seq 0 $((REPS - 1))); do
    W="r${r}_s0"; mkdir -p "$OUT/$W"
    docker run --rm --name "g4ev_${NAME}_$W" --network none --gpus "\"device=$GPUS\"" "${COMMON[@]}" "${MOUNTS[@]}" \
      -v "$OUT/$W:/kaggle/working/run" "$IMAGE" python /opt/run_eval.py --out /kaggle/working/run "${EXTRA[@]}" \
      2>&1 | tee "$OUT/$W/log.txt"
  done
  fix_owner
  python3 "$ROOT/eval/passk.py" "$OUT" | tee "$OUT/summary.txt"
  exit 0
fi

# ---------------- endpoint mode ----------------
NET="g4ev_${NAME}"; GW="g4gw_${NAME}"
UPSTREAM=$ENDPOINT; GW_HOST=()
if [[ "$ENDPOINT" =~ ^(https?://)(localhost|127\.0\.0\.1)([:/].*)?$ ]]; then  # endpoint on this host
  UPSTREAM="${BASH_REMATCH[1]}host.docker.internal${BASH_REMATCH[3]}"; GW_HOST=(--add-host host.docker.internal:host-gateway)
fi
cleanup() {
  docker ps -q --filter "name=g4ev_${NAME}_" | xargs -r docker stop -t 5 >/dev/null 2>&1 || true
  docker rm -f "$GW" >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
  fix_owner
}
trap cleanup EXIT
docker network create --internal "$NET" >/dev/null
mkdir -p "$OUT/model_logs"
GW_ARGS=(--upstream "$UPSTREAM" --backend "$BACKEND" --log-dir /logs)
[[ -n "$MODEL" ]] && GW_ARGS+=(--model "$MODEL")
[[ -n "$EXTRA_BODY" ]] && GW_ARGS+=(--extra-body "$EXTRA_BODY")
UPSTREAM_API_KEY="$API_KEY" docker run -d --name "$GW" -e UPSTREAM_API_KEY "${GW_HOST[@]}" \
  -v "$ROOT/eval/gateway.py:/gw.py:ro" -v "$OUT/model_logs:/logs" "$IMAGE" python -I /gw.py "${GW_ARGS[@]}" >/dev/null
docker network connect "$NET" "$GW"
for _ in $(seq 30); do docker exec "$GW" python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/')" 2>/dev/null && break; sleep 1; done
docker logs "$GW"
echo "workers: $REPS reps x $WORKERS shards, sandbox=$SANDBOX, out=$OUT"

pids=()
for r in $(seq 0 $((REPS - 1))); do for s in $(seq 0 $((WORKERS - 1))); do
  W="r${r}_s$s"; mkdir -p "$OUT/$W"
  docker run --rm --name "g4ev_${NAME}_$W" --network "$NET" "${COMMON[@]}" "${MOUNTS[@]}" \
    -v "$OUT/$W:/kaggle/working/run" "$IMAGE" python /opt/run_eval.py --out /kaggle/working/run "${EXTRA[@]}" \
    --shard "$s/$WORKERS" --external-endpoint "http://$GW:8000/w/$W/v1" --external-model "${MODEL:-model}" \
    > "$OUT/$W/log.txt" 2>&1 &
  pids+=($!); sleep 2
done; done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
[[ $fail -gt 0 ]] && echo "$fail worker(s) exited non-zero (see $OUT/*/log.txt)"
docker exec "$GW" python -c "import urllib.request; print('gateway', urllib.request.urlopen('http://127.0.0.1:8000/stats').read().decode())" || true
fix_owner
python3 "$ROOT/eval/passk.py" "$OUT" | tee "$OUT/summary.txt"
