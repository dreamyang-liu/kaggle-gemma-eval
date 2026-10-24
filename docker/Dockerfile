# NOTE: v170 = Python 3.12.13 / torch 2.10.0+cu128, matching competition notebooks (v171 is py3.13, incompatible).
# Replicates the Kaggle scoring notebook environment:
# Kaggle GPU image + competition wheelhouse installed with --no-deps --force-reinstall,
# exactly as in the official "Getting Started: Gemma 4 Developer Agent" notebook.
ARG BASE=gcr.io/kaggle-gpu-images/python:v170
FROM ${BASE}

COPY wheelhouse /tmp/wheelhouse_src
RUN set -eux; \
    rm -f /usr/local/lib/python*/dist-packages/*cutlass*.pth /usr/local/lib/python*/site-packages/*cutlass*.pth; \
    mkdir -p /tmp/wheelhouse; \
    for w in /tmp/wheelhouse_src/*.whl; do \
      n=$(basename "$w"); case "$n" in *cutlass*) continue;; esac; \
      case "$n" in *+cu128*) ;; *cu128*) n=$(echo "$n" | sed 's/cu128/+cu128/');; esac; \
      cp "$w" "/tmp/wheelhouse/$n"; \
    done; \
    python -m pip install -q --no-deps --force-reinstall /tmp/wheelhouse/*.whl; \
    rm -rf /tmp/wheelhouse /tmp/wheelhouse_src /root/.cache/pip

ENV TRANSFORMERS_NO_TF=1 TRANSFORMERS_NO_FLAX=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENGINE_READY_TIMEOUT_S=1200 \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 NCCL_DEBUG=WARN \
    LITELLM_LOCAL_MODEL_COST_MAP=True LITELLM_TELEMETRY=False LITELLM_LOG=WARNING \
    OPENAI_API_KEY=EMPTY LOCAL_API_KEY=EMPTY PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OTEL_SDK_DISABLED=true OTEL_PYTHON_DISABLED=true
WORKDIR /kaggle/working
