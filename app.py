import streamlit as st
import numpy as np
import time

from numba import njit, prange

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except Exception:
    OPENCL_AVAILABLE = False


st.title("⚡ HeteroCompute Lab 2.0 — CPU vs GPU Parallel Explorer")
st.caption("FI9070 — Heterogeneous Computing | Real-time Demo")

st.markdown("""
This interactive lab compares:

1️⃣ CPU (Single Thread — baseline)  
2️⃣ CPU (Parallel — OpenMP-like using Numba)  
3️⃣ GPU (OpenCL Kernel)

No datasets. No models. Only raw computation.
""")

n = st.slider("Select Matrix Size (n x n)", 150, 1300, 500, 50)

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
    end = time.time()

    return end - start


def run_cpu_parallel(n):
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)

    start = time.time()
    _ = matmul_parallel(A, B)
    end = time.time()

    return end - start


# ---------------- GPU OPENCL ----------------
def run_gpu(n):
    if not OPENCL_AVAILABLE:
        raise RuntimeError("OpenCL not available on this system.")

    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    C = np.zeros((n, n), dtype=np.float32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

    kernel = """
    __kernel void matmul(const int N,
                         __global float* A,
                         __global float* B,
                         __global float* C)
    {
        int row = get_global_id(0);
        int col = get_global_id(1);
        float sum = 0.0;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
    """

    program = cl.Program(ctx, kernel).build()

    start = time.time()

    program.matmul(
        queue,
        (n, n),
        None,
        np.int32(n),
        A_buf,
        B_buf,
        C_buf
    )

    cl.enqueue_copy(queue, C, C_buf)
    queue.finish()

    end = time.time()

    return end - start, ctx.devices[0].name


col1, col2, col3 = st.columns(3)

results = {}

if col1.button("Run CPU — Single Thread"):
    t = run_cpu_single(n)
    st.success(f"🧠 CPU Single: {t:.4f} sec")
    results["CPU Single"] = t

if col2.button("Run CPU — Parallel"):
    t = run_cpu_parallel(n)
    st.success(f"🚀 CPU Parallel: {t:.4f} sec")
    results["CPU Parallel"] = t

if col3.button("Run GPU (OpenCL)"):
    try:
        t, dev = run_gpu(n)
        st.success(f"🎮 GPU: {t:.4f} sec")
        st.info(f"Device: {dev}")
        results["GPU"] = t
    except Exception as e:
        st.error(str(e))

st.divider()

if results:
    st.subheader("📊 Speed Comparison")
    st.bar_chart(results)
