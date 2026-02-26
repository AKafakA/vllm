# vLLM Emulator 成本边界定义

本文档定义 vLLM emulator backend 中哪些组件**真实运行**（无建模）、哪些组件**需要建模**（仿真）。

## 类别 A: 真实运行（无需建模）

以下组件执行真实代码，产生真实 CPU 开销：

### API 入口
- `vllm/entrypoints/api_server.py` - FastAPI 服务器，HTTP 处理
- `vllm/entrypoints/openai/` - OpenAI 兼容 API 端点
- `vllm/entrypoints/grpc_server.py` - gRPC 服务器

### 分词与预处理
- `vllm/tokenizers/` - 分词器加载与推理
- `vllm/inputs/` - 输入预处理，提示词解析

### 调度策略
- `vllm/v1/engine/core.py` - 核心引擎，请求分发
- `vllm/v1/scheduler.py` - 调度逻辑，批次形成
- `vllm/v1/worker/` - Worker 生命周期管理

### 请求生命周期
- `vllm/engine/` - 引擎初始化，请求排队
- `vllm/sequence.py` - 序列管理
- `vllm/outputs.py` - 输出处理

### 后勤（低优先级）
- 日志、指标、统计收集

---

## 类别 B: 建模 - GPU 计算成本

以下组件使用查找表建模：

### Prefill 阶段
- **目标:** `vllm/v1/worker/gpu_worker.py` - 模型前向传播
- **建模:** T_prefill_gpu(prompt_len, batch_ctx, parallelism, dtype)
- **配置:** 按提示词 token 数分桶

### Decode 阶段
- **目标:** `vllm/v1/worker/gpu_worker.py` - Token 生成
- **建模:** T_decode_gpu_per_token(active_seqs, kv_state, parallelism, dtype)
- **配置:** 按并发序列数分桶

---

## 类别 C: 建模 - CPU↔GPU 交互

以下组件使用交互成本建模：

### KV Cache 管理
- **目标:** `vllm/v1/worker/gpu_worker.py` - KV cache 操作
- **建模:** T_offload(bytes, direction, bw_eff, overlap_ratio, sync_penalty, concurrency)

### 内存操作
- **目标:** `vllm/device_allocator/` - 设备内存分配
- **建模:** CPU/GPU 内存拷贝操作

### 同步
- **目标:** CUDA 同步原语
- **建模:** 同步点，stream 事件

---

## 类别 D: 建模 - 网络（PD 分离）

以下组件为预填充-解码分离建模：

### 张量并行
- **目标:** `vllm/distributed/` - All-reduce 操作
- **建模:** T_all_reduce(op, bytes, world_size, topology)
- **配置:** 按拓扑分类（NVLink, PCIe, IB）

### 流水线并行
- **目标:** `vllm/distributed/` - Send/recv 操作
- **建模:** T_send_recv(op, bytes, src_dst, topology)

### PD KV 传输
- **目标:** 自定义 PD 调度逻辑
- **建模:** T_kv_transfer(bytes, concurrency, topology)

---

## 模块快速参考

| 模块路径 | 类别 | 说明 |
|----------|------|------|
| `entrypoints/api_server.py` | A | HTTP 服务器 |
| `entrypoints/openai/` | A | OpenAI API |
| `tokenizers/` | A | 分词器 |
| `inputs/` | A | 预处理 |
| `v1/engine/core.py` | A | 引擎核心 |
| `v1/scheduler.py` | A | 调度器 |
| `v1/worker/gpu_worker.py` | B | GPU 计算 |
| `device_allocator/` | C | 内存操作 |
| `distributed/` | D | 网络操作 |

---

## 实现说明

1. **钩子点:** 用 oracle 调用替换 GPU 执行路径
2. **配置分离:** 每个组件独立的 JSON 文件
3. **回退:** 无配置时使用真实执行
4. **验证:** 与真实 GPU 执行做 A/B 测试
