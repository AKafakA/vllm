#!/bin/bash
source /workspace/vllm-v18-env/bin/activate
ONLINE_DIR="/workspace/eval_results/RTX-3060-12GB/online"
REAL_TAG="split2d"

for TAG in fix_split2d_chain fix_split2d_pool fix_optA_chain fix_optA_pool; do
    echo "--- $TAG ---"
    for RATE in 1 4 8; do
        REAL="$ONLINE_DIR/${REAL_TAG}_r${RATE}_real.json"
        EMU="$ONLINE_DIR/${TAG}_r${RATE}_emu.json"
        if [ -f "$EMU" ] && [ -f "$REAL" ]; then
            python3 -c "
import json
e=json.load(open('$EMU'))
r=json.load(open('$REAL'))
tpot=(e['mean_tpot_ms']-r['mean_tpot_ms'])/r['mean_tpot_ms']*100
e2e=(e.get('mean_e2el_ms',0)-r.get('mean_e2el_ms',0))/r.get('mean_e2el_ms',1)*100
ttft=(e['mean_ttft_ms']-r['mean_ttft_ms'])/r['mean_ttft_ms']*100
print(f'  R=$RATE: TPOT={tpot:+.1f}% E2E={e2e:+.1f}% TTFT={ttft:+.1f}%')
"
        fi
    done
done
