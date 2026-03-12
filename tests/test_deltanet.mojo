# Test DeltaNet inference kernel
from math import sqrt, exp, log
from memory import UnsafePointer, memset_zero
from collections import List

from nvfp4.deltanet import (
    DeltaNetConfig,
    DeltaNetState,
    deltanet_forward,
    softplus,
    sigmoid,
    silu,
    rms_norm_inplace,
    matvec,
    matvec_transposed,
    dot,
    conv1d_step,
    DN_RMS_EPS,
)


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #

fn approx_eq(a: Float32, b: Float32, tol: Float32 = 1e-4) -> Bool:
    """Check approximate equality."""
    var diff = a - b
    if diff < 0.0:
        diff = -diff
    if diff <= tol:
        return True
    var denom = a if a > b else b
    if denom < 0.0:
        denom = -denom
    if denom > 0.0:
        return diff / denom <= tol
    return False


fn make_list(n: Int, val: Float32) -> List[Float32]:
    """Create a List[Float32] of size n filled with val."""
    var buf = List[Float32](capacity=n)
    buf.resize(n, val)
    return buf^


fn make_list_seq(n: Int, start: Float32 = 0.0, step: Float32 = 0.01) -> List[Float32]:
    """Create a List[Float32] with sequential values."""
    var buf = List[Float32](capacity=n)
    for i in range(n):
        buf.append(start + Float32(i) * step)
    return buf^


fn make_identity_like(rows: Int, cols: Int, scale: Float32 = 1.0) -> List[Float32]:
    """Create a matrix where M[i, i % cols] = scale, rest = 0."""
    var buf = List[Float32](capacity=rows * cols)
    buf.resize(rows * cols, 0.0)
    var p = buf.unsafe_ptr()
    for i in range(rows):
        p[i * cols + (i % cols)] = scale
    return buf^


# ===----------------------------------------------------------------------=== #
# Activation Function Tests
# ===----------------------------------------------------------------------=== #

fn test_softplus():
    print("Testing softplus:")
    var r0 = softplus(0.0)
    print("  softplus(0.0) =", r0, "(expected ~0.6931)")

    var r_large = softplus(30.0)
    print("  softplus(30.0) =", r_large, "(expected ~30.0)")

    var r_neg = softplus(-30.0)
    print("  softplus(-30.0) =", r_neg, "(expected ~0.0)")

    var r1 = softplus(1.0)
    print("  softplus(1.0) =", r1, "(expected ~1.3133)")


fn test_sigmoid():
    print("\nTesting sigmoid:")
    var r0 = sigmoid(0.0)
    print("  sigmoid(0.0) =", r0, "(expected 0.5)")

    var r_large = sigmoid(10.0)
    print("  sigmoid(10.0) =", r_large, "(expected ~1.0)")

    var r_neg = sigmoid(-10.0)
    print("  sigmoid(-10.0) =", r_neg, "(expected ~0.0)")


fn test_silu():
    print("\nTesting silu:")
    var r0 = silu(0.0)
    print("  silu(0.0) =", r0, "(expected 0.0)")

    var r1 = silu(1.0)
    print("  silu(1.0) =", r1, "(expected ~0.7311)")

    var r_neg = silu(-1.0)
    print("  silu(-1.0) =", r_neg, "(expected ~-0.2689)")


# ===----------------------------------------------------------------------=== #
# Linear Algebra Tests
# ===----------------------------------------------------------------------=== #

fn test_matvec():
    """Test matrix-vector multiply: y = A @ x."""
    print("\nTesting matvec:")
    # A = [[1, 2, 3], [4, 5, 6]], x = [1, 1, 1], y = [6, 15]
    var a_buf = List[Float32](capacity=6)
    a_buf.append(1.0); a_buf.append(2.0); a_buf.append(3.0)
    a_buf.append(4.0); a_buf.append(5.0); a_buf.append(6.0)
    var x_buf = List[Float32](capacity=3)
    x_buf.append(1.0); x_buf.append(1.0); x_buf.append(1.0)
    var y_buf = make_list(2, 0.0)

    matvec(a_buf.unsafe_ptr(), x_buf.unsafe_ptr(), y_buf.unsafe_ptr(), 2, 3)
    print("  [1,2,3; 4,5,6] @ [1,1,1] = [", y_buf[0], ",", y_buf[1], "] (expected [6, 15])")


fn test_matvec_transposed():
    """Test transposed matrix-vector: y = A^T @ x."""
    print("\nTesting matvec_transposed:")
    # A = [[1, 2], [3, 4], [5, 6]] (3x2), x = [1, 1, 1]
    # y = A^T @ x = [9, 12]
    var a_buf = List[Float32](capacity=6)
    a_buf.append(1.0); a_buf.append(2.0)
    a_buf.append(3.0); a_buf.append(4.0)
    a_buf.append(5.0); a_buf.append(6.0)
    var x_buf = List[Float32](capacity=3)
    x_buf.append(1.0); x_buf.append(1.0); x_buf.append(1.0)
    var y_buf = make_list(2, 0.0)

    matvec_transposed(a_buf.unsafe_ptr(), x_buf.unsafe_ptr(), y_buf.unsafe_ptr(), 3, 2)
    print("  A^T @ [1,1,1] = [", y_buf[0], ",", y_buf[1], "] (expected [9, 12])")


fn test_dot():
    print("\nTesting dot product:")
    var a_buf = List[Float32](capacity=3)
    a_buf.append(1.0); a_buf.append(2.0); a_buf.append(3.0)
    var b_buf = List[Float32](capacity=3)
    b_buf.append(4.0); b_buf.append(5.0); b_buf.append(6.0)
    var d = dot(a_buf.unsafe_ptr(), b_buf.unsafe_ptr(), 3)
    print("  [1,2,3] . [4,5,6] =", d, "(expected 32)")


fn test_rms_norm():
    print("\nTesting rms_norm_inplace:")
    var x_buf = List[Float32](capacity=4)
    x_buf.append(1.0); x_buf.append(2.0); x_buf.append(3.0); x_buf.append(4.0)
    var xp = x_buf.unsafe_ptr()
    # rms = sqrt((1+4+9+16)/4) = sqrt(7.5) ~ 2.7386
    rms_norm_inplace(xp, 4, DN_RMS_EPS)
    print("  rms_norm([1,2,3,4]):")
    print("    [", x_buf[0], ",", x_buf[1], ",", x_buf[2], ",", x_buf[3], "]")
    var rms_val = sqrt(Float32(7.5))
    print("    expected x[0] ~", 1.0 / rms_val, " got", x_buf[0])


# ===----------------------------------------------------------------------=== #
# Conv1d Test
# ===----------------------------------------------------------------------=== #

fn test_conv1d_step():
    """Test single-step causal conv1d."""
    print("\nTesting conv1d_step:")
    var channels = 2
    var kernel_size = 3
    var hist = kernel_size - 1

    # Conv buffer: [channels, hist] = [2, 2]
    var conv_buf = make_list(channels * hist, 0.0)
    var cbp = conv_buf.unsafe_ptr()
    cbp[0] = 0.5; cbp[1] = 1.0   # ch0: [oldest, newer]
    cbp[2] = 0.2; cbp[3] = 0.3   # ch1: [oldest, newer]

    # Weights: [channels, kernel_size] = [2, 3]
    var conv_w = make_list(channels * kernel_size, 0.0)
    var cwp = conv_w.unsafe_ptr()
    cwp[0] = 1.0; cwp[1] = 2.0; cwp[2] = 3.0  # ch0
    cwp[3] = 0.5; cwp[4] = 1.5; cwp[5] = 2.5  # ch1

    # Dummy bias (not used).
    var dummy = make_list(1, 0.0)

    # New input: [2.0, 4.0]
    var new_input = make_list(channels, 0.0)
    var nip = new_input.unsafe_ptr()
    nip[0] = 2.0; nip[1] = 4.0

    var result = make_list(channels, 0.0)

    conv1d_step(nip, cbp, cwp, result.unsafe_ptr(), channels, kernel_size, False, dummy.unsafe_ptr())

    # Expected ch0: 0.5*1.0 + 1.0*2.0 + 2.0*3.0 = 8.5
    # Expected ch1: 0.2*0.5 + 0.3*1.5 + 4.0*2.5 = 10.55
    print("  ch0:", result[0], "(expected 8.5)")
    print("  ch1:", result[1], "(expected 10.55)")

    # Buffer shifted: ch0=[1.0, 2.0], ch1=[0.3, 4.0]
    print("  buf: ch0=[", conv_buf[0], ",", conv_buf[1], "] (expected [1.0, 2.0])")
    print("  buf: ch1=[", conv_buf[2], ",", conv_buf[3], "] (expected [0.3, 4.0])")


# ===----------------------------------------------------------------------=== #
# DeltaNet Config Test
# ===----------------------------------------------------------------------=== #

fn test_config():
    print("\nTesting DeltaNetConfig:")
    var cfg = DeltaNetConfig()
    print("  d_model:", cfg.d_model, "(expected 2048)")
    print("  num_key_heads:", cfg.num_key_heads, "(expected 16)")
    print("  num_value_heads:", cfg.num_value_heads, "(expected 32)")
    print("  key_dim:", cfg.key_dim, "(expected 128)")
    print("  value_dim:", cfg.value_dim, "(expected 128)")
    print("  v_groups:", cfg.v_groups(), "(expected 2)")
    print("  state floats/layer:", cfg.total_state_floats(), "(expected 524288)")
    print("  total_qkv_dim:", cfg.total_qkv_dim(), "(expected 8192)")
    print("  z_dim:", cfg.z_dim(), "(expected 4096)")


# ===----------------------------------------------------------------------=== #
# DeltaNet State Test
# ===----------------------------------------------------------------------=== #

fn test_state_init():
    print("\nTesting DeltaNetState initialization:")
    var cfg = DeltaNetConfig()
    var state = DeltaNetState(cfg)

    var all_zero = True
    var total = cfg.total_state_floats()
    for i in range(total):
        if state.s_matrix[i] != 0.0:
            all_zero = False
            break
    print("  State zero-initialized:", all_zero, "(expected True)")
    print("  State size:", total, "floats =", total * 4, "bytes")


# ===----------------------------------------------------------------------=== #
# Small DeltaNet Forward Test
# ===----------------------------------------------------------------------=== #

fn test_forward_small():
    """Test DeltaNet forward with a small configuration."""
    print("\nTesting deltanet_forward (small config):")
    var cfg = DeltaNetConfig(
        d_model=8,
        num_key_heads=2,
        num_value_heads=2,
        key_dim=4,
        value_dim=4,
        conv_kernel=4,
        rms_eps=DN_RMS_EPS,
    )

    var d = cfg.d_model
    var qkv_dim = cfg.total_qkv_dim()  # 2*4 + 2*4 + 2*4 = 24
    var z_dim = cfg.z_dim()            # 2*4 = 8
    var nvh = cfg.num_value_heads      # 2
    var q_dim = cfg.q_dim()            # 8
    var k_dim = cfg.k_dim()            # 8
    var v_dim = cfg.v_dim()            # 8
    var conv_k = cfg.conv_kernel       # 4

    print("  Config: d=8, nkh=2, nvh=2, kd=4, vd=4, qkv_dim=", qkv_dim)

    # Weights.
    var w_qkv = make_identity_like(qkv_dim, d, 0.1)
    var w_z = make_identity_like(z_dim, d, 0.5)
    var w_b = make_list(nvh * d, 0.1)
    var w_a = make_list(nvh * d, 0.05)
    var w_out = make_identity_like(d, z_dim, 1.0)
    var w_alog = make_list(nvh, 0.0)
    var w_alog_p = w_alog.unsafe_ptr()
    w_alog_p[0] = 0.0; w_alog_p[1] = 0.5
    var w_dtbias = make_list(nvh, 0.0)
    var wdt_p = w_dtbias.unsafe_ptr()
    wdt_p[0] = 0.0; wdt_p[1] = 0.1

    # Conv weights: pass-through newest input.
    var w_conv_q = make_list(q_dim * conv_k, 0.0)
    var w_conv_k = make_list(k_dim * conv_k, 0.0)
    var w_conv_v = make_list(v_dim * conv_k, 0.0)
    var wcq_p = w_conv_q.unsafe_ptr()
    var wck_p = w_conv_k.unsafe_ptr()
    var wcv_p = w_conv_v.unsafe_ptr()
    for ch in range(q_dim):
        wcq_p[ch * conv_k + (conv_k - 1)] = 1.0
    for ch in range(k_dim):
        wck_p[ch * conv_k + (conv_k - 1)] = 1.0
    for ch in range(v_dim):
        wcv_p[ch * conv_k + (conv_k - 1)] = 1.0

    # No conv bias.
    var w_conv_bias_q = List[Float32]()
    var w_conv_bias_k = List[Float32]()
    var w_conv_bias_v = List[Float32]()

    # State.
    var state = DeltaNetState(cfg)

    # Input: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    var hidden = make_list_seq(d, start=0.1, step=0.1)
    var result = make_list(d, 0.0)

    # First forward pass.
    deltanet_forward(
        hidden, result,
        w_qkv, w_z, w_b, w_a, w_out,
        w_alog, w_dtbias,
        w_conv_q, w_conv_k, w_conv_v,
        w_conv_bias_q, w_conv_bias_k, w_conv_bias_v,
        state, cfg,
    )

    print("  Input:  [", end="")
    for i in range(d):
        if i > 0:
            print(", ", end="")
        print(hidden[i], end="")
    print("]")

    print("  Output: [", end="")
    for i in range(d):
        if i > 0:
            print(", ", end="")
        print(result[i], end="")
    print("]")

    var any_nonzero = False
    for i in range(d):
        if result[i] != 0.0:
            any_nonzero = True
            break
    print("  Output non-zero:", any_nonzero, "(expected True)")

    var state_updated = False
    for i in range(cfg.total_state_floats()):
        if state.s_matrix[i] != 0.0:
            state_updated = True
            break
    print("  State updated:", state_updated, "(expected True)")

    # Second step — output should differ (state changed).
    var result2 = make_list(d, 0.0)
    deltanet_forward(
        hidden, result2,
        w_qkv, w_z, w_b, w_a, w_out,
        w_alog, w_dtbias,
        w_conv_q, w_conv_k, w_conv_v,
        w_conv_bias_q, w_conv_bias_k, w_conv_bias_v,
        state, cfg,
    )

    var outputs_differ = False
    for i in range(d):
        if not approx_eq(result[i], result2[i]):
            outputs_differ = True
            break
    print("  Second step differs:", outputs_differ, "(expected True)")


# ===----------------------------------------------------------------------=== #
# O(1) Memory Property Test
# ===----------------------------------------------------------------------=== #

fn test_o1_memory():
    """Verify that DeltaNet state size is constant regardless of sequence length."""
    print("\nTesting O(1) memory property:")
    var cfg = DeltaNetConfig()
    var state_bytes = cfg.total_state_floats() * 4
    print("  State per layer:", state_bytes, "bytes (", state_bytes // 1024, "KB)")
    print("  State is FIXED for any sequence length (no KV cache growth)")
    print("  For 30 DeltaNet layers:", 30 * state_bytes // 1024, "KB total")
    var expected = cfg.num_value_heads * cfg.key_dim * cfg.value_dim * 4
    print("  Expected:", expected, "bytes, got:", state_bytes)


# ===----------------------------------------------------------------------=== #
# Main
# ===----------------------------------------------------------------------=== #

fn main():
    print("=" * 60)
    print("DeltaNet Inference Kernel Test Suite")
    print("=" * 60)

    test_softplus()
    test_sigmoid()
    test_silu()
    test_matvec()
    test_matvec_transposed()
    test_dot()
    test_rms_norm()
    test_conv1d_step()
    test_config()
    test_state_init()
    test_forward_small()
    test_o1_memory()

    print("\n" + "=" * 60)
    print("All DeltaNet tests completed!")
    print("=" * 60)
