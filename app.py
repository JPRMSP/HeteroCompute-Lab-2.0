```python
import streamlit as st
import numpy as np
import time

from numba import njit, prange
from numba import cuda


st.title("⚡ HeteroCompute Lab — CPU vs Parallel vs GPU (CUDA)")
st.caption("FI9070 — Heterogeneous Computing | Real-time Performance Demo")

st.markdown("""
This lab compares:

1️⃣ CPU — Sequential  
2️⃣ CPU — Parallel (threaded)  
3️⃣ GPU — CUDA Kernel

No datasets. No pretrained models — only raw computation.
""")

n = st.slider("Matrix size (n x n)", 150, 1200, 500, 50)

st.divider()

# ---------------- CPU SINGLE ----------------
@njit
def matmul_single(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


# ---------------- CPU PARALLEL ----------------
@njit(parallel=True)
def matmul_parallel(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float32)
    for i in prange(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


def run_cpu_single(n):
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)

    start = time.time()
    _ = matmul_single(A, B)
    return time.time() - start


def run_cpu_parallel(n):
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)

    start = time.time()
    _ = matmul_parallel(A, B)
    return time.time() - start


# ---------------- GPU CUDA ----------------
@cuda.jit
def matmul_cuda(A, B, C):
    row = cuda.grid(1)
    n = C.shape[0]

    if row < n:
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[row, k] * B[k, j]
            C[row, j] = s


def run_gpu(n):
    if not cuda.is_available():
        raise RuntimeError("CUDA GPU not available. In Colab: Runtime ➜ Change runtime type ➜ GPU")

    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    C = np.zeros((n, n), dtype=np.float32)

    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    dC = cuda.to_device(C)

    threads = 128
    blocks = (n + threads - 1) // threads

    start = time.time()
    matmul_cuda[blocks, threads](dA, dB, dC)
    cuda.synchronize()
    end = time.time()

    return end - start


col1, col2, col3 = st.columns(3)
results = {}

if col1.button("Run CPU — Single"):
    t = run_cpu_single(n)
    st.success(f"🧠 CPU Single: {t:.4f} sec")
    results["CPU Single"] = t

if col2.button("Run CPU — Parallel"):
    t = run_cpu_parallel(n)
    st.success(f"🚀 CPU Parallel: {t:.4f} sec")
    results["CPU Parallel"] = t

if col3.button("Run GPU — CUDA"):
    try:
        t = run_gpu(n)
        st.success(f"🎮 GPU CUDA: {t:.4f} sec")
        results["GPU CUDA"] = t
    except Exception as e:
        st.error(str(e))

st.divider()

if results:
    st.subheader("📊 Speed Comparison")
    st.bar_chart(results)

st.markdown("""
### 🧠 Viva Points
- Sequential vs parallel vs GPU kernels
- Thread-level parallelism vs massive data parallelism
- Why GPU wins for large workloads
- Overhead vs computation trade-off
""")
