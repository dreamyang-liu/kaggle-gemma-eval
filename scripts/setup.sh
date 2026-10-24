#!/usr/bin/env bash
# One-time setup: download the harness wheelhouse and competition data from Kaggle, build the eval images and
# write dev_tasks.jsonl.
#   scripts/setup.sh [--sandbox-image] [--models 31b,12b,e4b,e2b]
#     --sandbox-image  also build swebench-sandbox:latest (needed for evaluate.sh --sandbox docker)
#     --models         also download QAT weights from Hugging Face (only needed for local vLLM mode)
# Kaggle auth: KAGGLE_API_TOKEN, ~/.kaggle/access_token, or ~/.kaggle/kaggle.json. You must have accepted the
# competition rules on kaggle.com first.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
BUILD_SANDBOX=0 MODELS=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sandbox-image) BUILD_SANDBOX=1; shift;;
    --models) MODELS=$2; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

AUTH=()
if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then AUTH=(-H "Authorization: Bearer $KAGGLE_API_TOKEN")
elif [[ -f ~/.kaggle/access_token ]]; then AUTH=(-H "Authorization: Bearer $(tr -d '[:space:]' < ~/.kaggle/access_token)")
elif [[ -f ~/.kaggle/kaggle.json ]]; then
  AUTH=(-u "$(python3 -c "import json,os;d=json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')));print(d['username']+':'+d['key'])")")
else echo "no Kaggle credentials found" >&2; exit 2; fi
fetch() {  # fetch URL ZIP DEST
  [[ -e "$3/.done" ]] && { echo "$3 already present"; return; }
  echo "downloading $1"
  curl -fL --retry 3 "${AUTH[@]}" -o "$2" "$1"
  mkdir -p "$3"; python3 -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$2" "$3"
  rm -f "$2"; touch "$3/.done"
}
fetch https://www.kaggle.com/api/v1/datasets/download/metric/gemma-4-developer-agent-wheelhouse /tmp/g4_wheelhouse.zip wheelhouse
fetch https://www.kaggle.com/api/v1/competitions/data/download-all/gemma-4-developer-agent /tmp/g4_data.zip data
# the zips may wrap everything in one top-level directory
for d in wheelhouse data; do
  if [[ $(find "$d" -mindepth 1 -maxdepth 1 ! -name .done | wc -l) == 1 ]]; then
    sub=$(find "$d" -mindepth 1 -maxdepth 1 -type d); [[ -n "$sub" ]] && { mv "$sub"/* "$d"/; rmdir "$sub"; }
  fi
done
ls wheelhouse/*.whl >/dev/null && [[ -f data/tasks.jsonl ]]

docker build -t gemma4-kaggle:v170 -f docker/Dockerfile .
docker build -t gemma4-kaggle:v170-plus -f docker/Dockerfile.plus .
if [[ $BUILD_SANDBOX == 1 ]]; then
  ctx=$(mktemp -d "$ROOT/.sandbox_ctx.XXXX")  # Dockerfile.public copies imp.py/telnetlib.py (data/docker) and wheels/ (data/wheels)
  cp data/docker/imp.py data/docker/telnetlib.py "$ctx"/ && cp -al data/wheels "$ctx/wheels"
  docker build -t swebench-sandbox:latest -f data/docker/Dockerfile.public "$ctx"; rm -rf "$ctx"
fi

python3 - <<'EOF'
import json
ids = {l.strip() for l in open('dev_task_ids.txt') if l.strip()}
rows = [l for l in open('data/tasks.jsonl') if json.loads(l)['instance_id'] in ids]
open('dev_tasks.jsonl', 'w').writelines(rows)
print(f'dev_tasks.jsonl: {len(rows)} of {len(ids)} dev task ids found')
EOF

if [[ -n "$MODELS" ]]; then
  declare -A HF=([31b]=google/gemma-4-31B-it-qat-w4a16-ct [12b]=google/gemma-4-12B-it-qat-w4a16-ct
                 [e4b]=google/gemma-4-E4B-it-qat-w4a16-ct [e2b]=google/gemma-4-E2B-it-qat-w4a16-ct)
  for m in ${MODELS//,/ }; do
    repo=${HF[$m]:?unknown model $m}; dir="models/$(basename "$repo" | tr '[:upper:]' '[:lower:]')"
    python3 -m huggingface_hub.commands.huggingface_cli download "$repo" --local-dir "$dir" \
      || huggingface-cli download "$repo" --local-dir "$dir"
  done
fi
echo "setup done"
