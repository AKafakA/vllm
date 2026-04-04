#!/bin/bash
# Create minimal CUDA stub libraries for running GPU vLLM on CPU-only hosts.
# These stubs satisfy the dynamic linker — all actual CUDA calls are
# intercepted by the emulator hooks before reaching these libraries.
set -e

STUB_DIR="${HOME}/vllm-emulator/cuda_stubs"
mkdir -p "${STUB_DIR}"

echo "Creating CUDA stub libraries in ${STUB_DIR}..."

# Create a minimal C file that provides empty symbols
cat > /tmp/cuda_stub.c << 'EOF'
#include <stddef.h>
// Minimal CUDA stub — provides symbols that the vLLM C extensions
// reference, but the emulator hooks intercept all actual GPU calls
// before they reach this level.

// libcuda.so symbols (CUDA driver API)
int cuInit(unsigned int flags) { return 0; }
int cuDeviceGetCount(int* count) { if(count) *count = 1; return 0; }
int cuDeviceGet(int* device, int ordinal) { if(device) *device = 0; return 0; }
int cuDeviceGetName(char* name, int len, int dev) { return 0; }
int cuDeviceGetAttribute(int* pi, int attrib, int dev) { if(pi) *pi = 0; return 0; }
int cuDeviceTotalMem(size_t* bytes, int dev) { if(bytes) *bytes = 12884901888UL; return 0; }
int cuCtxCreate(void** pctx, unsigned int flags, int dev) { return 0; }
int cuCtxGetCurrent(void** pctx) { if(pctx) *pctx = (void*)1; return 0; }
int cuMemGetInfo(size_t* free, size_t* total) { if(free) *free=12884901888UL; if(total) *total=12884901888UL; return 0; }

// libnvidia-ml.so symbols (NVML)
int nvmlInit() { return 0; }
int nvmlDeviceGetCount(unsigned int* count) { if(count) *count = 1; return 0; }
int nvmlShutdown() { return 0; }

// libcudart.so symbols (CUDA runtime)
int cudaGetDevice(int* device) { if(device) *device = 0; return 0; }
int cudaGetDeviceCount(int* count) { if(count) *count = 1; return 0; }
int cudaSetDevice(int device) { return 0; }
int cudaDeviceSynchronize() { return 0; }
int cudaStreamSynchronize(void* stream) { return 0; }
int cudaMemGetInfo(size_t* free, size_t* total) { if(free) *free=12884901888UL; if(total) *total=12884901888UL; return 0; }
int cudaMalloc(void** devPtr, size_t size) { return 0; }
int cudaFree(void* devPtr) { return 0; }
EOF

# Compile as shared libraries
gcc -shared -o "${STUB_DIR}/libcuda.so.1" /tmp/cuda_stub.c -fPIC
gcc -shared -o "${STUB_DIR}/libnvidia-ml.so.1" /tmp/cuda_stub.c -fPIC
gcc -shared -o "${STUB_DIR}/libcudart.so.12" /tmp/cuda_stub.c -fPIC

# Create symlinks
ln -sf libcuda.so.1 "${STUB_DIR}/libcuda.so"
ln -sf libnvidia-ml.so.1 "${STUB_DIR}/libnvidia-ml.so"
ln -sf libcudart.so.12 "${STUB_DIR}/libcudart.so"

echo "Stub libraries created:"
ls -la "${STUB_DIR}"

echo ""
echo "Usage: LD_LIBRARY_PATH=${STUB_DIR}:\${LD_LIBRARY_PATH} python3 ..."
