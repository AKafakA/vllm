#!/bin/bash
# Profiling configuration — source this before running profiling scripts
# Adjust these knobs to control profiling density and coverage

# === Online profiling rates ===
# Low rates: capture tt=1-10 (few concurrent requests)
# Mid rates: capture tt=5-30
# High rates: capture tt=20-100+ (many concurrent, high contention)
# inf: max concurrency coverage
ONLINE_RATES="0.5 1 2 4 6 8 10 12 16 20 24 32 48 64 inf"

# Prompts per rate (more = more data points, but takes longer)
# Low rates need fewer (arrival dominates), high rates need more
PROMPTS_LOW=100    # for rate <= 2
PROMPTS_MID=200    # for rate 4-12
PROMPTS_HIGH=500   # for rate 16+
PROMPTS_INF=500    # for rate=inf

# === Warmup ===
WARMUP_PROMPTS=200
WARMUP_RATE=4

# === Offline profiling ===
# Batch sizes for bench throughput
OFFLINE_BATCH_SIZES="50 100 200 300"
# Pure decode batch sizes (input=1 token)
OFFLINE_DECODE_SIZES="50 100 200"

# === Helper ===
get_prompts_for_rate() {
    local rate=$1
    if [ "$rate" = "inf" ]; then echo $PROMPTS_INF
    elif [ "$rate" = "0.5" ] || [ "$rate" = "1" ] || [ "$rate" = "2" ]; then echo $PROMPTS_LOW
    elif [ "$(echo "$rate <= 12" | bc)" = "1" ]; then echo $PROMPTS_MID
    else echo $PROMPTS_HIGH
    fi
}

echo "=== Profiling Configuration ==="
echo "Online rates: $ONLINE_RATES"
echo "Warmup: $WARMUP_PROMPTS prompts at rate=$WARMUP_RATE"
total_prompts=0
for rate in $ONLINE_RATES; do
    np=$(get_prompts_for_rate $rate)
    total_prompts=$((total_prompts + np))
done
echo "Total online prompts: ~$total_prompts"
echo "Offline batch sizes: $OFFLINE_BATCH_SIZES"
echo "Offline decode sizes: $OFFLINE_DECODE_SIZES"
