#!/bin/bash
#SBATCH --job-name=vllm-emu-eval
#SBATCH --account=KALYVIANAKI-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/wd312/hpc-work/vllm-emulator/slurm_%j.log

# Submit: sbatch slurm_eval.sh <model> <label> [tp]
# Example: sbatch slurm_eval.sh Qwen/Qwen2.5-7B-Instruct 7b-tp1 1

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
LABEL="${2:-1.5b-tp1}"
TP="${3:-1}"

echo "SLURM Job ${SLURM_JOB_ID} started on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Model: ${MODEL}, Label: ${LABEL}, TP: ${TP}"

# Run the evaluation
bash /home/wd312/Code/llm/vllm-emulator/paper/isca-2026-mlarchsys/scripts/csd3/run_eval.sh \
    "${MODEL}" "${LABEL}" "${TP}"

echo "SLURM Job ${SLURM_JOB_ID} completed"
