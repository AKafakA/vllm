#!/bin/bash
# Analyze worker subfunction timing + step-cycle exec_ms/sample_ms
# Outputs summary for Option 1 implementation
source /workspace/vllm-v18-env/bin/activate
export HF_HOME=/workspace/.hf_home HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub HF_HUB_OFFLINE=1

RESULT_DIR="/workspace/eval_results/RTX-3060-12GB"
TRACE="${RESULT_DIR}/step_cycle_fresh.jsonl"
WORKER_LOG="/workspace/worker_profile_srv.log"

echo "=== Worker Subfunction Summary ==="
echo ""
echo "Raw WorkerTiming entries (excluding step=1 warmup):"
grep "WorkerTiming" "$WORKER_LOG" 2>/dev/null | grep -v "step=1 " | while read line; do
    # Extract fields
    step=$(echo "$line" | grep -oP 'step=\K[0-9]+')
    tt=$(echo "$line" | grep -oP 'tt=\K[0-9]+')
    update=$(echo "$line" | grep -oP 'update=\K[0-9.]+')
    prep=$(echo "$line" | grep -oP 'prep=\K[0-9.]+')
    batch=$(echo "$line" | grep -oP 'batch=\K[0-9.]+')
    slots=$(echo "$line" | grep -oP 'slots=\K[0-9.]+')
    attn=$(echo "$line" | grep -oP 'attn=\K[0-9.]+')
    preproc=$(echo "$line" | grep -oP 'preproc=\K[0-9.]+')
    # Compute total using awk
    total=$(echo "$update $prep $batch $slots $attn $preproc" | awk '{printf "%.2f", $1+$2+$3+$4+$5+$6}')
    echo "  step=$step tt=$tt total=${total}ms (update=$update prep=$prep batch=$batch slots=$slots attn=$attn preproc=$preproc)"
done

echo ""
echo "=== Step-cycle trace exec_ms/sample_ms analysis ==="
echo ""
echo "Extracting exec_ms and sample_ms from $TRACE..."
# Use grep + awk to extract fields without Python
grep '"exec_ms"' "$TRACE" | grep -v '__marker__' | head -2000 | while read line; do
    tt=$(echo "$line" | grep -oP '"total_tokens":\s*\K[0-9]+')
    exec_ms=$(echo "$line" | grep -oP '"exec_ms":\s*\K[0-9.]+')
    sample_ms=$(echo "$line" | grep -oP '"sample_ms":\s*\K[0-9.]+')
    sched_ms=$(echo "$line" | grep -oP '"sched_ms":\s*\K[0-9.]+')
    sc_us=$(echo "$line" | grep -oP '"step_cycle_us":\s*\K[0-9.]+')
    if [ -n "$tt" ] && [ -n "$exec_ms" ] && [ -n "$sample_ms" ]; then
        echo "$tt $exec_ms $sample_ms $sched_ms $sc_us"
    fi
done > /tmp/exec_sample_data.txt

total_lines=$(wc -l < /tmp/exec_sample_data.txt)
echo "Total data points with exec_ms: $total_lines"

echo ""
echo "Averages by tt range:"
for range_label in "1-5" "6-15" "16-35" "36-100" "100+"; do
    case "$range_label" in
        "1-5")   filter='$1>=1 && $1<=5' ;;
        "6-15")  filter='$1>=6 && $1<=15' ;;
        "16-35") filter='$1>=16 && $1<=35' ;;
        "36-100") filter='$1>=36 && $1<=100' ;;
        "100+")  filter='$1>100' ;;
    esac
    awk "$filter"' {
        n++; sum_exec+=$2; sum_sample+=$3; sum_sched+=$4; sum_sc+=$5
    } END {
        if(n>0) printf "  tt=%-8s: exec=%.2fms sample=%.2fms sched=%.2fms step_cycle=%.2fms (n=%d)\n",
            "'"$range_label"'", sum_exec/n, sum_sample/n, sum_sched/n, sum_sc/n/1000, n
    }' /tmp/exec_sample_data.txt
done

echo ""
echo "=== Key ratio: CPU subfunctions / exec_ms ==="
echo "Worker CPU subfunctions avg: ~1.4ms (from WorkerTiming)"
echo "If exec_ms avg is X, then CPU fraction = 1.4 / X"
echo ""
echo "=== DONE $(date) ==="
