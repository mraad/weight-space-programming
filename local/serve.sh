#!/usr/bin/env bash
# Usage on gc1: local/serve.sh qwen | nemotron | stop | status
# Each server runs in its own window of tmux session "wsp", with a log in /opt/dlami/nvme/wsp/logs. No tool parser: the runs offer no tools.
set -uo pipefail
WSP=/opt/dlami/nvme/wsp
VLLM="$WSP/vllm-venv/bin/vllm"
ENVS="HF_HUB_CACHE=/opt/dlami/nvme/hf-hub PATH=$WSP/vllm-venv/bin:/usr/local/cuda/bin:$PATH"  # FlashInfer compiles its kernels with ninja and nvcc
MAX_LEN=${MAX_LEN:-262144}

window() {  # name, GPU list, command
  tmux has-session -t wsp 2>/dev/null || tmux new-session -d -s wsp -n main -c "$WSP"
  tmux new-window -t wsp -n "$1" -c "$WSP" "env $ENVS CUDA_VISIBLE_DEVICES=$2 $3 2>&1 | tee logs/$1.log; exec bash"
}

case "${1:-}" in
  qwen)  # one copy per card; 56 GB of weights leave room for the context on each 96 GB card
    for gpu in 0 1 2 3; do
      window "qwen$gpu" "$gpu" "$VLLM serve Qwen/Qwen3.8-27B --port 800$((gpu + 1)) --host 127.0.0.1 --max-model-len $MAX_LEN --reasoning-parser qwen3 --gpu-memory-utilization 0.92 --max-num-seqs ${SEQS:-8}"
      # The first server compiles the FlashInfer kernels into a shared cache; the others must not compile them at the same time.
      if [ "$gpu" = 0 ]; then
        for _ in $(seq 1 90); do [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/health)" = 200 ] && break; sleep 10; done
        [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/health)" = 200 ] || { echo "first server did not start: see logs/qwen0.log" >&2; exit 1; }
      fi
    done ;;
  nemotron)  # one copy over all 4 cards. The weights take about 62 GB of each card. The flags come from the model card, without its tool flags. vLLM has this architecture built in, so it needs no remote code.
    window nemotron 0,1,2,3 "$VLLM serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 --port 8001 --host 127.0.0.1 --tensor-parallel-size 4 --enable-expert-parallel --async-scheduling --dtype auto --kv-cache-dtype fp8 --max-model-len $MAX_LEN --gpu-memory-utilization 0.9 --max-num-seqs ${SEQS:-8} --max-cudagraph-capture-size 128 --enable-chunked-prefill --mamba-ssm-cache-dtype float32 --reasoning-parser nemotron_v3" ;;
  stop)
    for name in qwen0 qwen1 qwen2 qwen3 nemotron; do tmux kill-window -t "wsp:$name" 2>/dev/null; done
    pkill -f "$VLLM serve" 2>/dev/null
    true ;;
  status)
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    for port in 8001 8002 8003 8004; do printf '%s %s\n' "$port" "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health")"; done ;;
  *) echo "usage: $0 qwen | nemotron | stop | status" >&2; exit 2 ;;
esac
