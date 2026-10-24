"""Local replica of the official Kaggle "Getting Started: Gemma 4 Developer Agent" notebook.

Cells 2 (data/agent), 4 (vLLM server) and 5 (Phase 1 + Phase 2 via swegemma.Evaluator) are
reproduced as-is; only paths, task selection, sandbox type, L4 memory emulation and an optional
external model endpoint are parameterised. Runs inside the gemma4-kaggle image (see evaluate.sh),
with Kaggle-style mounts:
  /kaggle/input/competitions/gemma-4-developer-agent   competition data
  /kaggle/input/models/google/gemma-4/other/<model>/2   model weights
  /kaggle/agent                                         submission dir (read-only, copied)
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

for name in ('LiteLLM', 'LiteLLM Router', 'LiteLLM Proxy', 'litellm', 'httpx', 'httpcore',
             'openai', 'urllib3', 'opentelemetry', 'opentelemetry.context'):
    lg = logging.getLogger(name)
    lg.setLevel(logging.WARNING)
    if 'litellm' in name.lower():
        lg.propagate = False
        lg.handlers.clear()

# Kaggle L4: 23034 MiB visible per GPU; notebook uses gpu_memory_utilization=0.90 of that.
L4_TOTAL_MIB = 23034

ap = argparse.ArgumentParser()
ap.add_argument('--tasks', nargs='*', help='instance_ids to run (default: all)')
ap.add_argument('--first', type=int, help='only run the first N tasks')
ap.add_argument('--shard', default='0/1', help='i/n: run every n-th task starting at i')
ap.add_argument('--tasks-file', default=None, help='override tasks.jsonl path')
ap.add_argument('--out', default='/kaggle/working/run', help='output dir')
ap.add_argument('--port', type=int, default=8000)
ap.add_argument('--no-l4-emulation', action='store_true',
                help='use 0.90 of real GPU memory instead of emulating 4x L4 KV-cache size')
ap.add_argument('--external-endpoint', default=None,
                help='skip vLLM; route the declared model alias to this OpenAI-compatible /v1 endpoint')
ap.add_argument('--external-model', default=None, help='served model name at --external-endpoint')
ap.add_argument('--sandbox', choices=['subprocess', 'docker'], default='subprocess',
                help='subprocess = Kaggle notebook (agent commands run in this container); '
                     'docker = one sibling container per task (needs /var/run/docker.sock)')
ap.add_argument('--sandbox-image', default='swebench-sandbox:latest', help='image for --sandbox docker')
args = ap.parse_args()

import torch
import litellm
import pandas as pd
import yaml
from adk_submission import VllmConfig, VllmServer, discover_adapters
from adk_submission.discovery import discover_declared_models
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks

litellm.drop_params = True

# ---- Cell 2: data + agent ----
DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
AGENT_DIR = OUT / 'agent'
if AGENT_DIR.exists():
    shutil.rmtree(AGENT_DIR)
shutil.copytree('/kaggle/agent', AGENT_DIR)

TASKS_PATH = Path(args.tasks_file) if args.tasks_file else DATA_DIR / 'tasks.jsonl'
tasks = load_tasks(TASKS_PATH)
if args.tasks:
    want = set(args.tasks)
    tasks = [t for t in tasks if t.instance_id in want]
if args.first:
    tasks = tasks[:args.first]
si, sn = map(int, args.shard.split('/'))
tasks = tasks[si::sn]
print(f'Running {len(tasks)} tasks; agent={AGENT_DIR} sandbox={args.sandbox}', flush=True)

GRAPH_DIR = str(DATA_DIR / 'graphs')
EMBEDDINGS_DIR = str(DATA_DIR / 'embeddings')

# ---- Cell 4: vLLM server ----
ALLOWED_MODEL_NAMES = frozenset({
    'gemma-4-31b-it-qat-w4a16-ct',
    'gemma-4-12b-it-qat-w4a16-ct',
    'gemma-4-e4b-it-qat-w4a16-ct',
    'gemma-4-e2b-it-qat-w4a16-ct',
})
MODEL_VLLM_EXTRA_ARGS = {
    'gemma-4-12b-it-qat-w4a16-ct': (
        '--hf-overrides',
        '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}',
    ),
}
_SAFE_PATH_COMPONENT = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')
INFERENCE_API_KEY = os.environ.get('EVAL_API_KEY') or os.environ.get('LOCAL_API_KEY', 'EMPTY')

declared_models = discover_declared_models(AGENT_DIR)
assert len(declared_models) == 1, f'Expected exactly 1 declared model, got: {sorted(declared_models)}'
declared_model = next(iter(declared_models))
assert declared_model in ALLOWED_MODEL_NAMES, f'Unsupported model {declared_model!r}'
if args.external_endpoint:
    MODEL_PATH = None
else:
    MODEL_PATH = Path(f'/kaggle/input/models/google/gemma-4/other/{declared_model}/2')
    assert MODEL_PATH.exists(), f'Model weights not found at {MODEL_PATH}'

adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
adapter_map = getattr(adapters, 'adapters', None) or {}
for adapter_name in adapter_map:
    assert _SAFE_PATH_COMPONENT.fullmatch(adapter_name), f'Unsafe adapter name {adapter_name!r}'
detected_ranks = [r for info in adapter_map.values()
                  if (r := getattr(info, 'rank', None)) is not None and r > 0]

gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
gpu_util = 0.90
l4_args = ()
if not args.no_l4_emulation and torch.cuda.is_available():
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    gpu_util = round(0.90 * L4_TOTAL_MIB / total_mib, 4)
    # vLLM picks larger scheduler defaults on >=70 GiB GPUs; pin the <70 GiB (L4) API-server defaults.
    l4_args = ('--max-num-batched-tokens', '2048', '--max-num-seqs', '256')
print(f'tp={tp_size} gpu_memory_utilization={gpu_util} model={declared_model} '
      f'adapters={list(adapter_map)}', flush=True)

vllm_cfg = VllmConfig(
    model=str(MODEL_PATH),
    port=args.port,
    host='127.0.0.1',
    tool_call_parser='gemma4',
    reasoning_parser='gemma4',
    max_model_len=32768,
    dtype='bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 'auto',
    gpu_memory_utilization=gpu_util,
    enable_auto_tool_choice=True,
    enable_lora=bool(adapter_map),
    max_loras=max(1, len(adapter_map)),
    max_lora_rank=max(detected_ranks) if detected_ranks else 128,
    scheduler_reserve_full_isl=False,
    tensor_parallel_size=tp_size,
    startup_timeout=60 * 20,
    # L4 has no NVLink/P2P so vLLM falls back to NCCL there; on NVLink hosts custom all-reduce
    # also crashes under expandable_segments. Disable it to match Kaggle.
    extra_args=(*MODEL_VLLM_EXTRA_ARGS.get(declared_model, ()), '--disable-custom-all-reduce', *l4_args),
)
if args.external_endpoint:
    # Same fallback the scorer uses when attaching to an already-active endpoint.
    from adk_submission import ModelRegistry
    from google.adk.models.lite_llm import LiteLlm
    server_instance = None
    models = ModelRegistry()
    models.register(declared_model, LiteLlm(model=f'openai/{args.external_model}', api_base=args.external_endpoint,
                                            api_key=INFERENCE_API_KEY, num_retries=3))
    print(f'external model {args.external_model} @ {args.external_endpoint} as {declared_model}', flush=True)
else:
    server_instance = VllmServer(vllm_cfg, adapter_manifest=adapters)
    t0 = time.time()
    server_instance.start()
    print(f'vLLM up on {server_instance.base_url} in {time.time() - t0:.0f}s', flush=True)
    models = server_instance.create_model_registry(
        aliases=[declared_model], model_prefix='openai/', api_key=INFERENCE_API_KEY)

# ---- Cell 5: Phase 1 + Phase 2 ----
eval_config_file = next(
    (c for name in ('eval_config.yaml', 'eval_config.yml') if (c := AGENT_DIR / name).is_file()), None)
raw_eval_cfg = (yaml.safe_load(eval_config_file.read_text(encoding='utf-8')) or {}
                if eval_config_file is not None else {})
eval_section = raw_eval_cfg.get('evaluation', raw_eval_cfg)
timeout_seconds = int(eval_section.get('timeout_seconds', 300))
verification_timeout_seconds = int(eval_section.get('verification_timeout_seconds', 300))
max_tool_calls = int(eval_section.get('max_tool_calls', 100))
max_time_minutes = float(eval_section.get('max_time_minutes', 60.0))
turns_raw = eval_section.get('max_turns', eval_section.get('max_llm_calls'))
max_turns = int(turns_raw) if turns_raw is not None else None

limits, gen_constraints = build_submission_limits()
eval_config = EvalConfig(
    tasks_path=TASKS_PATH,
    snapshots_dir=DATA_DIR / 'snapshots',
    results_dir=OUT / 'results',
    submission_dir=AGENT_DIR,
    models=models,
    sandbox=args.sandbox,
    image=args.sandbox_image,
    timeout_seconds=timeout_seconds,
    verification_timeout_seconds=verification_timeout_seconds,
    max_time_minutes=max_time_minutes,
    max_tool_calls=max_tool_calls,
    max_turns=max_turns,
    limits=limits,
    generation_constraints=gen_constraints,
    adapter_manifest=adapters,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
    graph_dir=GRAPH_DIR,
    embeddings_dir=EMBEDDINGS_DIR,
    wheels_dir=DATA_DIR / 'wheels',
    verbose=False,
)
evaluator = Evaluator(eval_config)

predictions, rows = [], []
run_t0 = time.time()
try:
    for idx, task in enumerate(tasks, start=1):
        print(f'[{idx}/{len(tasks)}] {task.instance_id} ({task.repo})', flush=True)
        result = asyncio.run(evaluator.evaluate_task(task=task, task_index=idx, total_tasks=len(tasks)))
        patch = (result.agent_patch or '').strip()
        predictions.append({'id': task.instance_id, 'prediction': patch if patch else 'NO_PATCH'})
        row = dict(id=task.instance_id, resolved=bool(result.resolved), exit_code=result.test_exit_code,
                   patch_chars=len(patch), tool_calls=result.tool_calls,
                   duration_s=round(result.duration_seconds, 1))
        rows.append(row)
        print(f'  -> {row}', flush=True)
        with open(OUT / 'results.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
finally:
    if server_instance is not None:
        server_instance.stop()

pd.DataFrame(predictions, columns=['id', 'prediction']).to_parquet(OUT / 'submission.parquet', index=False)
n_ok = sum(r['resolved'] for r in rows)
summary = dict(model=args.external_model or declared_model, n=len(rows), resolved=n_ok,
               score=round(n_ok / max(1, len(rows)), 4), wall_min=round((time.time() - run_t0) / 60, 1))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
print('SUMMARY', json.dumps(summary), flush=True)
