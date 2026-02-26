# Component Review: Network (Category D)

**Date:** 2026-02-26
**Component:** Distributed Communication
**Category:** D - Network (TP/PP/PD)

---

## 1. Overview

Network operations in vLLM are used for:
1. **Tensor Parallelism (TP)** - All-reduce for layer outputs
2. **Pipeline Parallelism (PP)** - Send/recv between stages
3. **Prefill-Decode (PD) Separation** - KV transfer between prefill and decode nodes

---

## 2. Key Components

### 2.1 Device Communicators (`distributed/device_communicators/`)

| File | Purpose |
|------|---------|
| `cuda_communicator.py` | NCCL-based communication |
| `custom_all_reduce.py` | Custom all-reduce implementations |
| `pynccl.py` | PyNCCL wrapper |
| `base_device_communicator.py` | Base interface |

### 2.2 KV Transfer (`distributed/kv_transfer/`)

| File | Purpose |
|------|---------|
| `kv_connector/v1/base.py` | KV connector interface |
| `mooncake_connector.py` | MoonCake KV transfer |
| `lmcache_connector.py` | LMCache KV transfer |
| `offloading_connector.py` | Offloading-based KV transfer |

---

## 3. Hook Points

### 3.1 All-Reduce (Tensor Parallelism)

**File:** `cuda_communicator.py`
**Method:** `all_reduce()` (line 130)

```python
def all_reduce(self, input_):
    # Multiple implementations:
    # 1. vllm.all_reduce_symmetric_with_copy
    # 2. Quick all-reduce
    # 3. Custom all-reduce
    # 4. Symmetric memory all-reduce
    # 5. PyNCCL all-reduce
    # 6. PyTorch distributed all-reduce
```

**Variability:** High
- Different implementations for different configs
- Topology-dependent (NVLink vs PCIe vs IB)

### 3.2 Send/Recv (Pipeline Parallelism)

**File:** `cuda_communicator.py`
**Methods:** 
- `send()` (line 240)
- `recv()` (line 252)

```python
def send(self, tensor, dst):
    # Either PyNCCL send or torch.distributed.send

def recv(self, size, dtype, src):
    # Either PyNCCL recv or torch.distributed.recv
```

**Variability:** Medium
- Used in PP stages
- Can be blocking or async

### 3.3 KV Transfer (PD Separation)

**File:** `kv_connector/v1/base.py`
**Key Methods:**

| Method | Purpose |
|--------|---------|
| `start_load_kv()` | Start loading KV from remote |
| `save_kv_layer()` | Save KV to remote |
| `wait_for_layer_load()` | Wait for load to complete |
| `wait_for_save()` | Wait for save to complete |
| `get_finished()` | Get completed transfers |

**Variability:** High
- Different connectors (MoonCake, LMCache, etc.)
- Different protocols (RDMA, TCP, etc.)

---

## 4. Emulation Strategy

### Option 1: Oracle Hook at Communicator Level

```python
# In cuda_communicator.py
def all_reduce(self, input_):
    # Before real all-reduce
    latency = oracle.get_all_reduce_latency(
        bytes=input_.numel() * input_.element_size(),
        world_size=self.world_size,
        topology=self.get_topology()  # NVLink/PCIe/IB
    )
    
    if emulated_mode:
        time.sleep(latency / 1e6)
        return input_
    
    # Real all-reduce
    return self._do_real_all_reduce(input_)
```

### Option 2: Oracle Hook at KV Connector Level

```python
# In KVConnectorBase
def start_load_kv(self, forward_context, **kwargs):
    latency = oracle.get_kv_transfer_latency(
        bytes=self.calculate_bytes(**kwargs),
        direction="load",
        topology=self.get_topology()
    )
    
    if emulated_mode:
        # Fake async transfer
        return
    
    # Real load
    return self._do_real_load_kv(forward_context, **kwargs)
```

---

## 5. Profile Schema

```json
{
  "all_reduce": {
    "nvlink": [
      {"bytes": 4096, "world_size": 2, "latency_us": 50},
      {"bytes": 1048576, "world_size": 2, "latency_us": 200},
      {"bytes": 1048576, "world_size": 8, "latency_us": 800}
    ],
    "pcie": [
      {"bytes": 4096, "world_size": 2, "latency_us": 100},
      {"bytes": 1048576, "world_size": 2, "latency_us": 500}
    ],
    "ib": [
      {"bytes": 4096, "world_size": 2, "latency_us": 80},
      {"bytes": 1048576, "world_size": 2, "latency_us": 300}
    ]
  },
  "send_recv": {
    "nvlink": [
      {"bytes": 4096, "latency_us": 30},
      {"bytes": 1048576, "latency_us": 150}
    ],
    "tcp": [
      {"bytes": 4096, "latency_us": 200},
      {"bytes": 1048576, "latency_us": 1000}
    ]
  },
  "kv_transfer": {
    "rdma": [
      {"bytes": 4096, "concurrency": 1, "latency_us": 50},
      {"bytes": 4096, "concurrency": 8, "latency_us": 350}
    ],
    "tcp": [
      {"bytes": 4096, "concurrency": 1, "latency_us": 150},
      {"bytes": 4096, "concurrency": 8, "latency_us": 800}
    ]
  }
}
```

**Dimensions:**
- `bytes` - Transfer size
- `world_size` - Number of GPUs (for all-reduce)
- `topology` - NVLink/PCIe/IB (network topology)
- `concurrency` - Number of concurrent transfers

---

## 6. Task Definition

### Task P1.3: Network Cost Model (Category D)

**Objective:** Create oracle for network operations

**Deliverables:**
1. `vllm_emulator/oracle/network_cost_oracle.py`
   - `get_all_reduce_latency(bytes, world_size, topology)`
   - `get_send_latency(bytes, topology)`
   - `get_recv_latency(bytes, topology)`
   - `get_kv_transfer_latency(bytes, direction, topology, concurrency)`
2. Hook into `cuda_communicator.all_reduce()` (conditional)
3. Hook into `KVConnectorBase` methods
4. Per-topology profile packs

**Acceptance Criteria:**
- [ ] Oracle interface defined
- [ ] Can load topology-specific profiles
- [ ] All-reduce latency = f(bytes, world_size, topology)
- [ ] Send/Recv latency = f(bytes, topology)
- [ ] KV transfer latency = f(bytes, direction, topology, concurrency)
- [ ] Can fallback to real execution

---

## 7. Notes

- **Topology matters:** NVLink is much faster than PCIe/IB
- **Concurrency degrades performance:** Multiple concurrent transfers slow down
- **Multiple implementations:** Each communicator has several all-reduce paths
- **PD is optional:** KV transfer is only used in disaggregated serving
- **Reference:** Vidur profiles by topology (a100_pair_nvlink, a100_dgx, etc.)
