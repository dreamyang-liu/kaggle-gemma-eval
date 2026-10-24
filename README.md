# gemma4-agent-eval

Local evaluation for the Kaggle competition [Google – The Gemma 4 Developer Agent](https://www.kaggle.com/competitions/gemma-4-developer-agent).
It runs the official harness (`swegemma` / `adk-submission` from the competition wheelhouse) in a replica of the
Kaggle scoring image. You point it at any OpenAI-compatible model endpoint, or let it serve the QAT model locally
with vLLM.

## Setup

```bash
scripts/setup.sh                    # wheelhouse + competition data (Kaggle API), images, dev_tasks.jsonl
scripts/setup.sh --sandbox-image    # also build swebench-sandbox:latest for --sandbox docker
scripts/setup.sh --models 31b       # also download QAT weights (local vLLM mode only)
```

Setup needs docker (plus the NVIDIA runtime for local mode) and Kaggle credentials: `KAGGLE_API_TOKEN`,
`~/.kaggle/access_token` or `~/.kaggle/kaggle.json`. Accept the competition rules on kaggle.com first. The data and
weights are downloaded, not stored in this repo.

## Run

```bash
# any OpenAI-compatible endpoint (self-hosted vLLM/SGLang, hosted APIs); the key can also come from EVAL_API_KEY
./evaluate.sh --agent agents/shell_v2 --endpoint http://localhost:8000/v1 --model gemma4 --workers 8

# Gemma 4 31B on OpenRouter, pinned to bf16 providers (others serve fp8/fp4)
EVAL_API_KEY=$OPENROUTER_API_KEY ./evaluate.sh --agent agents/shell_v2 \
  --endpoint https://openrouter.ai/api/v1 --backend openrouter --model google/gemma-4-31b-it \
  --extra-body '{"provider": {"quantizations": ["bf16"], "allow_fallbacks": false}}' --workers 32 --reps 2

# a stronger model as a teacher, e.g. for sampling trajectories
EVAL_API_KEY=$OPENROUTER_API_KEY ./evaluate.sh --agent agents/budget_v2 \
  --endpoint https://openrouter.ai/api/v1 --backend openrouter --model moonshotai/kimi-k3

# local: serve the agent's declared model with vLLM on 4 GPUs, emulating Kaggle's 4x L4 KV-cache budget
./evaluate.sh --agent agents/shell_v2 --gpus 0,1,2,3
```

| option | default | |
|---|---|---|
| `--agent DIR` | required | submission dir (`agent.yaml` at the root, same layout as `submission.zip`) |
| `--endpoint URL` | none, so local vLLM | OpenAI-compatible base URL ending in `/v1`; `localhost` works too |
| `--model NAME` | request's model | model name sent to the endpoint |
| `--api-key KEY` | `$EVAL_API_KEY` | only the gateway container sees it |
| `--backend` | `passthrough` | `passthrough` (vLLM/SGLang), `strict` (strict OpenAI APIs) or `openrouter` (see `eval/gateway.py`) |
| `--extra-body JSON` | none | merged into every upstream request, e.g. OpenRouter `provider` routing |
| `--sandbox` | `subprocess` | `subprocess` runs agent commands in the eval container, as in the Kaggle notebook; `docker` gives each task its own sibling container (`swebench-sandbox:latest`) |
| `--workers N` / `--reps R` | 16 / 1 | N parallel shards per rep (endpoint mode); reps are used for pass@k |
| `--tasks-file F` | `dev_tasks.jsonl` | `data/tasks.jsonl` holds all 129 public tasks |
| `--tasks ID ...`, `--first N` | all | subset |
| `--name NAME` | `<agent>_<time>` | output dir `runs/NAME` |

The image comes from `IMAGE` (default `gemma4-kaggle:v170-plus`), and the data and weights from `DATA_DIR` and
`MODELS_DIR`.

Output in `runs/NAME/`:
- `r<rep>_s<shard>/`: `results.jsonl`, `submission.parquet`, `results/traces/*.json` (ATIF trajectories) and `log.txt`
- `model_logs/<worker>.jsonl`: every model request/response, reasoning included
- `summary.txt`: pass@k, per-repo pass@1 and failure modes. Recompute it with `python3 eval/passk.py runs/A [runs/B ...]`.

## How it works

- `eval/run_eval.py` reproduces cells 2, 4 and 5 of the official Getting Started notebook: a vLLM server with the same
  arguments, then `swegemma.Evaluator` Phase 1 (agent) and Phase 2 (hidden tests), tasks in sequence. Only paths,
  task selection, the sandbox type, L4 emulation and the external endpoint are parameterised.
- Endpoint mode: the eval workers run on an internal docker network with no internet, as on Kaggle. `eval/gateway.py`
  is their only route out: it adds the API key, adapts the request for the upstream, retries 429/5xx responses and logs
  every call. The `openrouter` backend turns thinking on with `reasoning: {"enabled": true}` (Gemma 4 has it off by
  default there) and sends earlier reasoning back as the `reasoning_details` OpenRouter returned.
- A hosted Gemma 4 31B is not the Kaggle model: Kaggle serves the W4A16 QAT build with vLLM on 4x L4, while OpenRouter
  providers serve bf16/fp8/fp4 builds. Pin `quantizations` with `--extra-body` and treat scores as an approximation
  of local or Kaggle runs.
- Local mode reproduces Kaggle's 4x L4 KV-cache size (gpu_memory_utilization = 0.9 × 23034 MiB / local GPU memory,
  plus vLLM's scheduler defaults for GPUs under 70 GiB). Decode speed is not emulated, so wall-clock time on fast GPUs
  or hosted APIs is optimistic compared with Kaggle (about 8.7 s per model call on 4x L4 with thinking on).
- `gemma4-kaggle:v170` = Kaggle GPU image v170 + the competition wheelhouse installed with `--no-deps`, as the official
  notebook does. `v170-plus` adds `inline-snapshot` and `dirty-equals`, which some public fastapi tests import.
  `dev_task_ids.txt` lists the 106 of 129 public tasks whose gold patch passes in that image.
- With `--sandbox docker`, fastapi tasks fail to import in the public sandbox image (its wheels lack some test
  dependencies), so use `subprocess` for scores comparable to Kaggle.

## Agents

| agent | tools | budget | pass@1 / pass@2 on the dev set* |
|---|---|---|---|
| `agents/budget_v2` | 9 standard tools (prompt forbids the 3 graph tools) | 8 min / 28 calls | 0.396 / 0.472 (2 reps); 0.413 / 0.478 / pass@4 0.519 (4 reps) |
| `agents/shell_v2` | `run_command`, `get_status`, `submit_patch` | 8 min / 28 calls | 0.410 / 0.472 (2 reps) |

\* Gemma 4 31B through a hosted API, not the Kaggle QAT build; temperature 1.0, subprocess sandbox. With temperature
0.2, shell_v2 scored 0.410 / 0.471.

`agents/budget_v2` is the configuration from
[flexonafft/gemma-4-budget-aware-coding-agent-v2](https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent-v2)
(Apache-2.0) with temperature raised from 0.2 to 1.0. `agents/shell_v2` uses the same prompt with tool instructions
rewritten for shell-only use.

Package a submission:

```bash
docker run --rm -v $PWD:/w gemma4-kaggle:v170 python -I /w/scripts/package_submission.py /w/agents/shell_v2 /w/shell_v2.zip
```
