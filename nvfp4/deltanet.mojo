# ===----------------------------------------------------------------------=== #
# DeltaNet Inference Kernel for MAX
#
# Gated DeltaNet (linear attention with delta rule) for autoregressive
# single-token inference. Replaces standard attention with O(1) memory
# per step — no growing KV cache.
#
# Architecture (Qwen3.5-35B-A3B reference):
#   - 30 DeltaNet layers (every 4th layer is full attention)
#   - 16 key heads, 32 value heads (V_GROUPS=2)
#   - key_dim=128, value_dim=128 per head
#   - State: num_value_heads x key_dim x value_dim per layer
#   - Conv1d with kernel_size=4 on Q, K, V
#
# Recurrence (per head h):
#   decay = exp(-softplus(A_log[h]) * softplus(alpha[h] + dt_bias[h]))
#   S[h] = decay * S[h] + beta[h] * outer(k[h], v[h] - S[h]^T @ k[h])
#   o[h] = silu(z[h]) * rms_norm(S[h]^T @ q[h])
#
# Reference: Yang et al. 2024 "Gated Delta Networks"
# ===----------------------------------------------------------------------=== #

from math import exp, sqrt, log
from memory import UnsafePointer, memset_zero
from collections import List

from .config import (
    NVFP4_GROUP_SIZE,
    NVFP4_PACK_FACTOR,
    NvFp4Linear,
    FP4_E2M1_LUT,
    unpack_fp4_pair,
    dequant_fp4,
)


# ===----------------------------------------------------------------------=== #
# DeltaNet Configuration Constants
# ===----------------------------------------------------------------------=== #

# Qwen3.5-35B-A3B DeltaNet defaults.
comptime DN_NUM_KEY_HEADS: Int = 16
comptime DN_NUM_VALUE_HEADS: Int = 32
comptime DN_KEY_DIM: Int = 128
comptime DN_VALUE_DIM: Int = 128
comptime DN_V_GROUPS: Int = 2  # value_heads / key_heads
comptime DN_CONV_KERNEL: Int = 4  # Conv1d kernel size for Q, K, V.
comptime DN_RMS_EPS: Float32 = 1e-6


# ===----------------------------------------------------------------------=== #
# Activation Functions
# ===----------------------------------------------------------------------=== #

@always_inline
fn softplus(x: Float32) -> Float32:
    """Compute softplus: log(1 + exp(x))."""
    # Numerically stable: for large x, softplus(x) ~ x.
    if x > 20.0:
        return x
    if x < -20.0:
        return Float32(0.0)
    return log(1.0 + exp(x))


@always_inline
fn sigmoid(x: Float32) -> Float32:
    """Compute sigmoid: 1 / (1 + exp(-x))."""
    if x >= 0.0:
        var e = exp(-x)
        return 1.0 / (1.0 + e)
    else:
        var e = exp(x)
        return e / (1.0 + e)


@always_inline
fn silu(x: Float32) -> Float32:
    """Compute SiLU: x * sigmoid(x)."""
    return x * sigmoid(x)


# ===----------------------------------------------------------------------=== #
# DeltaNet Layer Configuration
# ===----------------------------------------------------------------------=== #

struct DeltaNetConfig(Copyable, Movable):
    """Configuration for a DeltaNet layer."""
    var d_model: Int           # Model dimension (input/output).
    var num_key_heads: Int     # Number of key/query heads.
    var num_value_heads: Int   # Number of value heads.
    var key_dim: Int           # Key dimension per head.
    var value_dim: Int         # Value dimension per head.
    var conv_kernel: Int       # Conv1d kernel size.
    var rms_eps: Float32       # RMS norm epsilon.

    fn __init__(out self):
        """Default config matching Qwen3.5-35B-A3B DeltaNet layers."""
        self.d_model = 2048
        self.num_key_heads = DN_NUM_KEY_HEADS
        self.num_value_heads = DN_NUM_VALUE_HEADS
        self.key_dim = DN_KEY_DIM
        self.value_dim = DN_VALUE_DIM
        self.conv_kernel = DN_CONV_KERNEL
        self.rms_eps = DN_RMS_EPS

    fn __init__(
        out self,
        d_model: Int,
        num_key_heads: Int,
        num_value_heads: Int,
        key_dim: Int,
        value_dim: Int,
        conv_kernel: Int,
        rms_eps: Float32,
    ):
        self.d_model = d_model
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.conv_kernel = conv_kernel
        self.rms_eps = rms_eps

    fn __copyinit__(out self, copy: Self):
        self.d_model = copy.d_model
        self.num_key_heads = copy.num_key_heads
        self.num_value_heads = copy.num_value_heads
        self.key_dim = copy.key_dim
        self.value_dim = copy.value_dim
        self.conv_kernel = copy.conv_kernel
        self.rms_eps = copy.rms_eps

    fn __moveinit__(out self, deinit take: Self):
        self.d_model = take.d_model
        self.num_key_heads = take.num_key_heads
        self.num_value_heads = take.num_value_heads
        self.key_dim = take.key_dim
        self.value_dim = take.value_dim
        self.conv_kernel = take.conv_kernel
        self.rms_eps = take.rms_eps

    fn v_groups(self) -> Int:
        """Value heads per key head group."""
        return self.num_value_heads // self.num_key_heads

    fn state_size_per_head(self) -> Int:
        """State matrix size per head: key_dim * value_dim."""
        return self.key_dim * self.value_dim

    fn total_state_floats(self) -> Int:
        """Total state floats per layer: num_value_heads * key_dim * value_dim."""
        return self.num_value_heads * self.key_dim * self.value_dim

    fn total_qkv_dim(self) -> Int:
        """Total QKV projection output dimension."""
        return (
            self.num_key_heads * self.key_dim
            + self.num_key_heads * self.key_dim
            + self.num_value_heads * self.value_dim
        )

    fn q_dim(self) -> Int:
        return self.num_key_heads * self.key_dim

    fn k_dim(self) -> Int:
        return self.num_key_heads * self.key_dim

    fn v_dim(self) -> Int:
        return self.num_value_heads * self.value_dim

    fn z_dim(self) -> Int:
        """Output gate dimension: num_value_heads * value_dim."""
        return self.num_value_heads * self.value_dim


# ===----------------------------------------------------------------------=== #
# DeltaNet Recurrent State
# ===----------------------------------------------------------------------=== #

struct DeltaNetState:
    """Persistent recurrent state for a DeltaNet layer.

    State matrix S: [num_value_heads, key_dim, value_dim] in FP32.
    Conv1d buffer: sliding window of last (kernel_size - 1) projections
    for Q, K, V to support causal conv1d during autoregressive decode.
    """
    var s_matrix: List[Float32]      # [num_value_heads, key_dim, value_dim]
    var conv_buf_q: List[Float32]    # [q_dim, conv_kernel - 1]
    var conv_buf_k: List[Float32]    # [k_dim, conv_kernel - 1]
    var conv_buf_v: List[Float32]    # [v_dim, conv_kernel - 1]
    var config: DeltaNetConfig

    fn __init__(out self, config: DeltaNetConfig):
        self.config = config.copy()
        var state_size = config.total_state_floats()
        self.s_matrix = List[Float32](capacity=state_size)
        self.s_matrix.resize(state_size, 0.0)

        var conv_hist = config.conv_kernel - 1
        var q_buf_size = config.q_dim() * conv_hist
        var k_buf_size = config.k_dim() * conv_hist
        var v_buf_size = config.v_dim() * conv_hist

        self.conv_buf_q = List[Float32](capacity=q_buf_size)
        self.conv_buf_q.resize(q_buf_size, 0.0)
        self.conv_buf_k = List[Float32](capacity=k_buf_size)
        self.conv_buf_k.resize(k_buf_size, 0.0)
        self.conv_buf_v = List[Float32](capacity=v_buf_size)
        self.conv_buf_v.resize(v_buf_size, 0.0)

    fn state_offset(self, head: Int) -> Int:
        """Offset into s_matrix for a specific value head."""
        return head * self.config.key_dim * self.config.value_dim


# ===----------------------------------------------------------------------=== #
# Core Compute Primitives
# ===----------------------------------------------------------------------=== #

@always_inline
fn matvec_transposed[
    o_mat: Origin, o_vec: Origin, o_res: MutOrigin,
](
    mat: UnsafePointer[Float32, o_mat],   # [rows, cols] row-major
    vec: UnsafePointer[Float32, o_vec],   # [rows]
    result: UnsafePointer[Float32, o_res],# [cols]
    rows: Int,
    cols: Int,
):
    """Compute result = mat^T @ vec."""
    for c in range(cols):
        var acc = Float32(0.0)
        for r in range(rows):
            acc += (mat + r * cols + c)[] * (vec + r)[]
        (result + c).init_pointee_copy(acc)


@always_inline
fn matvec[
    o_mat: Origin, o_vec: Origin, o_res: MutOrigin,
](
    mat: UnsafePointer[Float32, o_mat],   # [rows, cols] row-major
    vec: UnsafePointer[Float32, o_vec],   # [cols]
    result: UnsafePointer[Float32, o_res],# [rows]
    rows: Int,
    cols: Int,
):
    """Compute result = mat @ vec."""
    for r in range(rows):
        var acc = Float32(0.0)
        for c in range(cols):
            acc += (mat + r * cols + c)[] * (vec + c)[]
        (result + r).init_pointee_copy(acc)


@always_inline
fn dot[o1: Origin, o2: Origin](
    a: UnsafePointer[Float32, o1],
    b: UnsafePointer[Float32, o2],
    n: Int,
) -> Float32:
    """Compute dot product of two vectors of length n."""
    var acc = Float32(0.0)
    for i in range(n):
        acc += (a + i)[] * (b + i)[]
    return acc


@always_inline
fn rms_norm_inplace[o: MutOrigin](
    x: UnsafePointer[Float32, o],
    n: Int,
    eps: Float32,
):
    """Apply RMS normalization in-place."""
    var sum_sq = Float32(0.0)
    for i in range(n):
        var val = (x + i)[]
        sum_sq += val * val
    var scale = 1.0 / sqrt(sum_sq / Float32(n) + eps)
    for i in range(n):
        var val = (x + i)[]
        (x + i).init_pointee_copy(val * scale)


@always_inline
fn conv1d_step[
    o_in: Origin, o_buf: MutOrigin, o_w: Origin, o_b: Origin, o_res: MutOrigin,
](
    new_input: UnsafePointer[Float32, o_in],
    conv_buf: UnsafePointer[Float32, o_buf],
    conv_weight: UnsafePointer[Float32, o_w],
    result: UnsafePointer[Float32, o_res],
    channels: Int,
    kernel_size: Int,
    has_bias: Bool,
    conv_bias: UnsafePointer[Float32, o_b],
):
    """Single-step causal conv1d for autoregressive inference.

    The conv buffer stores the last (kernel_size - 1) inputs.
    Weight layout: [channels, kernel_size] where weight[:, -1] applies to
    the newest input and weight[:, 0] to the oldest buffered input.

    After computing, shifts the buffer and inserts the new input.
    """
    var hist = kernel_size - 1

    for ch in range(channels):
        var acc = Float32(0.0)
        # Apply weights to buffered history (oldest first).
        for t in range(hist):
            acc += (conv_buf + ch * hist + t)[] * (conv_weight + ch * kernel_size + t)[]
        # Apply weight to current input (newest).
        acc += (new_input + ch)[] * (conv_weight + ch * kernel_size + hist)[]
        # Add bias if present.
        if has_bias:
            acc += (conv_bias + ch)[]
        (result + ch).init_pointee_copy(acc)

    # Shift buffer: drop oldest, append new input.
    for ch in range(channels):
        for t in range(hist - 1):
            var val = (conv_buf + ch * hist + t + 1)[]
            (conv_buf + ch * hist + t).init_pointee_copy(val)
        var new_val = (new_input + ch)[]
        (conv_buf + ch * hist + hist - 1).init_pointee_copy(new_val)


# ===----------------------------------------------------------------------=== #
# DeltaNet Forward (Single Token, Autoregressive)
# ===----------------------------------------------------------------------=== #

fn deltanet_forward(
    mut hidden_buf: List[Float32],     # Input: [d_model]
    mut output_buf: List[Float32],     # Output: [d_model] (written)
    mut w_qkv: List[Float32],         # [qkv_dim, d_model]
    mut w_z: List[Float32],           # [z_dim, d_model]
    mut w_b: List[Float32],           # [nvh, d_model]
    mut w_a: List[Float32],           # [nvh, d_model]
    mut w_out: List[Float32],         # [d_model, z_dim]
    mut w_alog: List[Float32],        # [nvh]
    mut w_dtbias: List[Float32],      # [nvh]
    mut w_conv_q: List[Float32],      # [q_dim, kernel]
    mut w_conv_k: List[Float32],      # [k_dim, kernel]
    mut w_conv_v: List[Float32],      # [v_dim, kernel]
    mut w_conv_bias_q: List[Float32], # [q_dim] or empty
    mut w_conv_bias_k: List[Float32], # [k_dim] or empty
    mut w_conv_bias_v: List[Float32], # [v_dim] or empty
    mut state: DeltaNetState,
    config: DeltaNetConfig,
):
    """DeltaNet single-token forward pass for autoregressive inference.

    Computes one step of the gated DeltaNet recurrence:
      1. Project input to Q, K, V, Z, beta, alpha
      2. Apply causal conv1d to Q, K, V
      3. Apply SiLU activation to Q, K, V (post-conv)
      4. For each value head group:
         a. Compute decay from A_log and alpha
         b. Compute delta: v_corrected = v - S^T @ k
         c. Update state: S = decay * S + beta * outer(k, v_corrected)
         d. Compute output: o = S^T @ q
      5. Apply output gate: o = silu(z) * rms_norm(o)
      6. Project back to d_model via out_proj
    """
    var d = config.d_model
    var nvh = config.num_value_heads
    var kd = config.key_dim
    var vd = config.value_dim
    var vg = config.v_groups()
    var qkv_dim = config.total_qkv_dim()
    var z_dim = config.z_dim()
    var conv_k = config.conv_kernel
    var q_dim = config.q_dim()
    var k_dim_total = config.k_dim()
    var v_dim_total = config.v_dim()
    var has_conv_bias = len(w_conv_bias_q) > 0

    # Get mutable pointers to all buffers.
    var hidden = hidden_buf.unsafe_ptr()
    var output_p = output_buf.unsafe_ptr()
    var wq = w_qkv.unsafe_ptr()
    var wz = w_z.unsafe_ptr()
    var wb = w_b.unsafe_ptr()
    var wa = w_a.unsafe_ptr()
    var wo = w_out.unsafe_ptr()
    var walog = w_alog.unsafe_ptr()
    var wdt = w_dtbias.unsafe_ptr()
    var wcq = w_conv_q.unsafe_ptr()
    var wck = w_conv_k.unsafe_ptr()
    var wcv = w_conv_v.unsafe_ptr()
    var s_ptr = state.s_matrix.unsafe_ptr()
    var cbq = state.conv_buf_q.unsafe_ptr()
    var cbk = state.conv_buf_k.unsafe_ptr()
    var cbv = state.conv_buf_v.unsafe_ptr()

    # ---------------------------------------------------------------
    # Step 1: Input projections
    # ---------------------------------------------------------------
    var qkv_buf = List[Float32](capacity=qkv_dim)
    qkv_buf.resize(qkv_dim, 0.0)
    var qkv_p = qkv_buf.unsafe_ptr()
    matvec(wq, hidden, qkv_p, qkv_dim, d)

    var z_buf = List[Float32](capacity=z_dim)
    z_buf.resize(z_dim, 0.0)
    var z_p = z_buf.unsafe_ptr()
    matvec(wz, hidden, z_p, z_dim, d)

    var beta_raw_buf = List[Float32](capacity=nvh)
    beta_raw_buf.resize(nvh, 0.0)
    var beta_raw = beta_raw_buf.unsafe_ptr()
    matvec(wb, hidden, beta_raw, nvh, d)

    var alpha_raw_buf = List[Float32](capacity=nvh)
    alpha_raw_buf.resize(nvh, 0.0)
    var alpha_raw = alpha_raw_buf.unsafe_ptr()
    matvec(wa, hidden, alpha_raw, nvh, d)

    # QKV split pointers (views into qkv_buf).
    var q_raw = qkv_p
    var k_raw = qkv_p + q_dim
    var v_raw = qkv_p + q_dim + k_dim_total

    # ---------------------------------------------------------------
    # Step 2: Causal conv1d on Q, K, V
    # ---------------------------------------------------------------
    var q_buf = List[Float32](capacity=q_dim)
    q_buf.resize(q_dim, 0.0)
    var q_p = q_buf.unsafe_ptr()

    var k_buf = List[Float32](capacity=k_dim_total)
    k_buf.resize(k_dim_total, 0.0)
    var k_p = k_buf.unsafe_ptr()

    var v_buf = List[Float32](capacity=v_dim_total)
    v_buf.resize(v_dim_total, 0.0)
    var v_p = v_buf.unsafe_ptr()

    # Dummy bias buffer (never read when has_bias=False, avoids aliasing).
    var dummy_bias = List[Float32](capacity=1)
    dummy_bias.resize(1, 0.0)
    var dummy_bp = dummy_bias.unsafe_ptr()

    if has_conv_bias:
        var wcbq = w_conv_bias_q.unsafe_ptr()
        var wcbk = w_conv_bias_k.unsafe_ptr()
        var wcbv = w_conv_bias_v.unsafe_ptr()
        conv1d_step(q_raw, cbq, wcq, q_p, q_dim, conv_k, True, wcbq)
        conv1d_step(k_raw, cbk, wck, k_p, k_dim_total, conv_k, True, wcbk)
        conv1d_step(v_raw, cbv, wcv, v_p, v_dim_total, conv_k, True, wcbv)
    else:
        conv1d_step(q_raw, cbq, wcq, q_p, q_dim, conv_k, False, dummy_bp)
        conv1d_step(k_raw, cbk, wck, k_p, k_dim_total, conv_k, False, dummy_bp)
        conv1d_step(v_raw, cbv, wcv, v_p, v_dim_total, conv_k, False, dummy_bp)

    # ---------------------------------------------------------------
    # Step 3: SiLU activation on Q, K, V (post-conv)
    # ---------------------------------------------------------------
    for i in range(q_dim):
        q_p[i] = silu(q_p[i])
    for i in range(k_dim_total):
        k_p[i] = silu(k_p[i])
    for i in range(v_dim_total):
        v_p[i] = silu(v_p[i])

    # ---------------------------------------------------------------
    # Step 4: Decay, beta, state update, output
    # ---------------------------------------------------------------
    var beta_buf = List[Float32](capacity=nvh)
    beta_buf.resize(nvh, 0.0)
    var beta_p = beta_buf.unsafe_ptr()
    for h in range(nvh):
        beta_p[h] = sigmoid(beta_raw[h])

    var decay_buf = List[Float32](capacity=nvh)
    decay_buf.resize(nvh, 0.0)
    var decay_p = decay_buf.unsafe_ptr()
    for h in range(nvh):
        var a_val = softplus(walog[h])
        var dt_val = softplus(alpha_raw[h] + wdt[h])
        decay_p[h] = exp(-a_val * dt_val)

    # Output accumulator.
    var o_buf = List[Float32](capacity=z_dim)
    o_buf.resize(z_dim, 0.0)
    var o_p = o_buf.unsafe_ptr()

    # Workspace buffers.
    var vc_buf = List[Float32](capacity=vd)
    vc_buf.resize(vd, 0.0)
    var vc_p = vc_buf.unsafe_ptr()

    var stk_buf = List[Float32](capacity=vd)
    stk_buf.resize(vd, 0.0)
    var stk_p = stk_buf.unsafe_ptr()

    # Per value-head recurrence.
    for h in range(nvh):
        var kh = h // vg  # Corresponding key head index.

        var q_h = q_p + kh * kd
        var k_h = k_p + kh * kd
        var v_h = v_p + h * vd
        var o_h = o_p + h * vd
        var s_h = s_ptr + state.state_offset(h)

        # 4a. Compute S^T @ k -> stk_p [value_dim]
        matvec_transposed(s_h, k_h, stk_p, kd, vd)

        # 4b. Delta correction: v_corrected = v - S^T @ k
        for i in range(vd):
            vc_p[i] = (v_h + i)[] - stk_p[i]

        # 4c. State update: S[h] = decay * S[h] + beta * outer(k, v_corrected)
        var d_h = decay_p[h]
        var b_h = beta_p[h]
        for r in range(kd):
            var k_r = (k_h + r)[]
            for c in range(vd):
                var idx = r * vd + c
                var old_val = (s_h + idx)[]
                (s_h + idx).init_pointee_copy(d_h * old_val + b_h * k_r * vc_p[c])

        # 4d. Output: o[h] = S^T @ q -> [value_dim]
        matvec_transposed(s_h, q_h, o_h, kd, vd)

    # ---------------------------------------------------------------
    # Step 5: Output gate — o = silu(z) * rms_norm(o)
    # ---------------------------------------------------------------
    for h in range(nvh):
        rms_norm_inplace(o_p + h * vd, vd, config.rms_eps)

    for i in range(z_dim):
        o_p[i] = silu(z_p[i]) * o_p[i]

    # ---------------------------------------------------------------
    # Step 6: Output projection — [d_model, z_dim] @ [z_dim] -> [d_model]
    # ---------------------------------------------------------------
    matvec(wo, o_p, output_p, d, z_dim)


# ===----------------------------------------------------------------------=== #
# NVFP4-Fused DeltaNet Matvec
# ===----------------------------------------------------------------------=== #

fn nvfp4_matvec[
    o_packed: Origin, o_scales: Origin, o_in: Origin, o_res: MutOrigin,
](
    linear: NvFp4Linear,
    packed_weights: UnsafePointer[UInt8, o_packed],
    blockscales: UnsafePointer[Scalar[DType.float8_e4m3fn], o_scales],
    input_vec: UnsafePointer[Float32, o_in],
    result: UnsafePointer[Float32, o_res],
):
    """NVFP4 matrix-vector product with fused dequantization.

    Computes: result = alpha * dequant(packed_weights) @ input
    For single-token inference (N=1 GEMV).
    """
    var M = linear.out_features
    var K = linear.in_features
    var alpha = linear.alpha()
    var packed_stride = K // NVFP4_PACK_FACTOR
    var scale_stride = K // NVFP4_GROUP_SIZE

    for m in range(M):
        var acc = Float32(0.0)
        for k in range(K):
            var packed_k = k // NVFP4_PACK_FACTOR
            var packed_byte = (packed_weights + m * packed_stride + packed_k)[]
            var lo_hi = unpack_fp4_pair(packed_byte)

            var scale_k = k // NVFP4_GROUP_SIZE
            var scale = (blockscales + m * scale_stride + scale_k)[].cast[DType.float32]()

            var fp4_val = lo_hi[0] if (k % 2 == 0) else lo_hi[1]
            var weight = dequant_fp4(fp4_val, scale)

            acc += weight * (input_vec + k)[]

        (result + m).init_pointee_copy(acc * alpha)
