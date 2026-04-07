#!/bin/bash
# Sync corrected vllm core files to proper locations on CloudLab
# Run ON the CloudLab host after rsync'ing to /tmp/vllm_sync/
set -e
CODE_DIR="${HOME}/vllm-emulator/code"

cp /tmp/vllm_sync/uniproc_executor.py "${CODE_DIR}/vllm/v1/executor/uniproc_executor.py"
cp /tmp/vllm_sync/core.py "${CODE_DIR}/vllm/v1/engine/core.py"
cp /tmp/vllm_sync/gpu_worker.py "${CODE_DIR}/vllm/v1/worker/gpu_worker.py"
echo "Core files copied to ${CODE_DIR}"

# Verify the executor hook has sched_delay
grep -c "sched_delay" "${CODE_DIR}/vllm_emulator/hooks/executor_hook.py" && echo "Executor hook: sched_delay present"
grep -c "cuda_graph_warmup" "${CODE_DIR}/vllm_emulator/hooks/executor_hook.py" && echo "Executor hook: CUDA graph warmup present"
