# Emulator Examples

This folder contains example assets and templates used by vLLM. For emulator‑specific examples, use the profile packs and run vLLM with emulator flags.

## Quick emulator run
```bash
python -m vllm.entrypoints.openai.api_server \
  --model <your-model> \
  --emulator-mode online \
  --profile-pack examples/profiles/a100-sxm-80gb.json
```

## Profile packs
- `examples/profiles/*.json`

## Notes
- Online mode blocks for estimated latency (serving‑like behavior).
- Offline mode does not sleep (batch analysis).
