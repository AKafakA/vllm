#!/bin/bash
pkill -9 -f chain_ttft_arrival 2>/dev/null
pkill -9 -f smoke_arrival 2>/dev/null
pkill -9 -f "bench serve" 2>/dev/null
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "vllm.entrypoints" 2>/dev/null
fuser 8100/tcp 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 10
ps aux | grep -E "vllm|bench serve" | grep -v grep | wc -l
