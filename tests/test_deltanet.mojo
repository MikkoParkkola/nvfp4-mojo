# Test DeltaNet inference kernel.
from std.math import sqrt
from std.collections import List
from std.testing import assert_equal, assert_true

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
from tests.helpers import expect_close


def make_list(n: Int, val: Float32) -> List[Float32]:
    """Create a List[Float32] of size n filled with val."""
    var buf = List[Float32](capacity=n)
    buf.resize(n, val)
    return buf^


def make_list_seq(
    n: Int, start: Float32 = 0.0, step: Float32 = 0.01
) -> List[Float32]:
    """Create a List[Float32] with sequential values."""
    var buf = List[Float32](capacity=n)
    for i in range(n):
        buf.append(start + Float32(i) * step)
    return buf^


def make_identity_like(
    rows: Int, cols: Int, scale: Float32 = 1.0
) -> List[Float32]:
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


def test_softplus() raises:
    expect_close(softplus(0.0), 0.6931472, "softplus(0.0)")
    expect_close(softplus(30.0), 30.0, "softplus(30.0)")
    expect_close(softplus(-30.0), 0.0, "softplus(-30.0)")
    expect_close(softplus(1.0), 1.3132616, "softplus(1.0)")
    print("  PASS softplus")


def test_sigmoid() raises:
    expect_close(sigmoid(0.0), 0.5, "sigmoid(0.0)")
    expect_close(sigmoid(10.0), 0.9999546, "sigmoid(10.0)")
    expect_close(sigmoid(-10.0), 4.539787e-05, "sigmoid(-10.0)")
    print("  PASS sigmoid")


def test_silu() raises:
    expect_close(silu(0.0), 0.0, "silu(0.0)")
    expect_close(silu(1.0), 0.7310586, "silu(1.0)")
    expect_close(silu(-1.0), -0.26894143, "silu(-1.0)")
    print("  PASS silu")


# ===----------------------------------------------------------------------=== #
# Linear Algebra Tests
# ===----------------------------------------------------------------------=== #


def test_matvec() raises:
    """Test matrix-vector multiply: y = A @ x."""
    var a_buf = List[Float32](capacity=6)
    a_buf.append(1.0)
    a_buf.append(2.0)
    a_buf.append(3.0)
    a_buf.append(4.0)
    a_buf.append(5.0)
    a_buf.append(6.0)
    var x_buf = List[Float32](capacity=3)
    x_buf.append(1.0)
    x_buf.append(1.0)
    x_buf.append(1.0)
    var y_buf = make_list(2, 0.0)

    matvec(a_buf.unsafe_ptr(), x_buf.unsafe_ptr(), y_buf.unsafe_ptr(), 2, 3)
    expect_close(y_buf[0], 6.0, "matvec y[0]")
    expect_close(y_buf[1], 15.0, "matvec y[1]")
    print("  PASS matvec")


def test_matvec_transposed() raises:
    """Test transposed matrix-vector: y = A^T @ x."""
    var a_buf = List[Float32](capacity=6)
    a_buf.append(1.0)
    a_buf.append(2.0)
    a_buf.append(3.0)
    a_buf.append(4.0)
    a_buf.append(5.0)
    a_buf.append(6.0)
    var x_buf = List[Float32](capacity=3)
    x_buf.append(1.0)
    x_buf.append(1.0)
    x_buf.append(1.0)
    var y_buf = make_list(2, 0.0)

    matvec_transposed(
        a_buf.unsafe_ptr(), x_buf.unsafe_ptr(), y_buf.unsafe_ptr(), 3, 2
    )
    expect_close(y_buf[0], 9.0, "matvec_transposed y[0]")
    expect_close(y_buf[1], 12.0, "matvec_transposed y[1]")
    print("  PASS matvec_transposed")


def test_dot() raises:
    var a_buf = List[Float32](capacity=3)
    a_buf.append(1.0)
    a_buf.append(2.0)
    a_buf.append(3.0)
    var b_buf = List[Float32](capacity=3)
    b_buf.append(4.0)
    b_buf.append(5.0)
    b_buf.append(6.0)
    var d = dot(a_buf.unsafe_ptr(), b_buf.unsafe_ptr(), 3)
    expect_close(d, 32.0, "dot")
    print("  PASS dot")


def test_rms_norm() raises:
    var x_buf = List[Float32](capacity=4)
    x_buf.append(1.0)
    x_buf.append(2.0)
    x_buf.append(3.0)
    x_buf.append(4.0)
    var xp = x_buf.unsafe_ptr()
    rms_norm_inplace(xp, 4, DN_RMS_EPS)
    var rms_val = sqrt(Float32(7.5))
    expect_close(x_buf[0], 1.0 / rms_val, "rms_norm x[0]")
    expect_close(x_buf[1], 2.0 / rms_val, "rms_norm x[1]")
    expect_close(x_buf[2], 3.0 / rms_val, "rms_norm x[2]")
    expect_close(x_buf[3], 4.0 / rms_val, "rms_norm x[3]")
    print("  PASS rms_norm")


# ===----------------------------------------------------------------------=== #
# Conv1d Test
# ===----------------------------------------------------------------------=== #


def test_conv1d_step() raises:
    """Test single-step causal conv1d."""
    var channels = 2
    var kernel_size = 3
    var hist = kernel_size - 1

    var conv_buf = make_list(channels * hist, 0.0)
    var cbp = conv_buf.unsafe_ptr()
    cbp[0] = 0.5
    cbp[1] = 1.0  # ch0: [oldest, newer]
    cbp[2] = 0.2
    cbp[3] = 0.3  # ch1: [oldest, newer]

    var conv_w = make_list(channels * kernel_size, 0.0)
    var cwp = conv_w.unsafe_ptr()
    cwp[0] = 1.0
    cwp[1] = 2.0
    cwp[2] = 3.0  # ch0
    cwp[3] = 0.5
    cwp[4] = 1.5
    cwp[5] = 2.5  # ch1

    var dummy = make_list(1, 0.0)

    var new_input = make_list(channels, 0.0)
    var nip = new_input.unsafe_ptr()
    nip[0] = 2.0
    nip[1] = 4.0

    var result = make_list(channels, 0.0)

    conv1d_step(
        nip,
        cbp,
        cwp,
        result.unsafe_ptr(),
        channels,
        kernel_size,
        False,
        dummy.unsafe_ptr(),
    )

    expect_close(result[0], 8.5, "conv1d_step ch0")
    expect_close(result[1], 10.55, "conv1d_step ch1")
    expect_close(conv_buf[0], 1.0, "conv1d_step buffer ch0[0]")
    expect_close(conv_buf[1], 2.0, "conv1d_step buffer ch0[1]")
    expect_close(conv_buf[2], 0.3, "conv1d_step buffer ch1[0]")
    expect_close(conv_buf[3], 4.0, "conv1d_step buffer ch1[1]")
    print("  PASS conv1d_step")


# ===----------------------------------------------------------------------=== #
# DeltaNet Config Test
# ===----------------------------------------------------------------------=== #


def test_config() raises:
    var cfg = DeltaNetConfig()
    assert_equal(
        cfg.d_model, 2048, "DeltaNetConfig d_model changed unexpectedly"
    )
    assert_equal(
        cfg.num_key_heads,
        16,
        "DeltaNetConfig num_key_heads changed unexpectedly",
    )
    assert_equal(
        cfg.num_value_heads,
        32,
        "DeltaNetConfig num_value_heads changed unexpectedly",
    )
    assert_equal(
        cfg.key_dim, 128, "DeltaNetConfig key_dim changed unexpectedly"
    )
    assert_equal(
        cfg.value_dim, 128, "DeltaNetConfig value_dim changed unexpectedly"
    )
    assert_equal(
        cfg.v_groups(), 2, "DeltaNetConfig v_groups changed unexpectedly"
    )
    assert_equal(
        cfg.total_state_floats(),
        524288,
        "DeltaNetConfig total_state_floats changed unexpectedly",
    )
    assert_equal(
        cfg.total_qkv_dim(),
        8192,
        "DeltaNetConfig total_qkv_dim changed unexpectedly",
    )
    assert_equal(cfg.z_dim(), 4096, "DeltaNetConfig z_dim changed unexpectedly")
    print("  PASS config")


# ===----------------------------------------------------------------------=== #
# DeltaNet State Test
# ===----------------------------------------------------------------------=== #


def test_state_init() raises:
    var cfg = DeltaNetConfig()
    var state = DeltaNetState(cfg)

    var all_zero = True
    var total = cfg.total_state_floats()
    for i in range(total):
        if state.s_matrix[i] != 0.0:
            all_zero = False
            break
    assert_true(all_zero, "DeltaNetState should zero-initialize s_matrix")
    assert_equal(total, 524288, "DeltaNetState size changed unexpectedly")
    assert_equal(
        total * 4, 2097152, "DeltaNetState byte size changed unexpectedly"
    )
    print("  PASS state_init")


# ===----------------------------------------------------------------------=== #
# Small DeltaNet Forward Test
# ===----------------------------------------------------------------------=== #


def test_forward_small() raises:
    """Test DeltaNet forward with a small configuration."""
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
    var qkv_dim = cfg.total_qkv_dim()
    var z_dim = cfg.z_dim()
    var nvh = cfg.num_value_heads
    var q_dim = cfg.q_dim()
    var k_dim = cfg.k_dim()
    var v_dim = cfg.v_dim()
    var conv_k = cfg.conv_kernel

    assert_equal(qkv_dim, 24, "small DeltaNet qkv dim changed unexpectedly")
    assert_equal(z_dim, 8, "small DeltaNet z dim changed unexpectedly")
    assert_equal(nvh, 2, "small DeltaNet num_value_heads changed unexpectedly")
    assert_equal(q_dim, 8, "small DeltaNet q dim changed unexpectedly")
    assert_equal(k_dim, 8, "small DeltaNet k dim changed unexpectedly")
    assert_equal(v_dim, 8, "small DeltaNet v dim changed unexpectedly")

    var w_qkv = make_identity_like(qkv_dim, d, 0.1)
    var w_z = make_identity_like(z_dim, d, 0.5)
    var w_b = make_list(nvh * d, 0.1)
    var w_a = make_list(nvh * d, 0.05)
    var w_out = make_identity_like(d, z_dim, 1.0)
    var w_alog = make_list(nvh, 0.0)
    var w_alog_p = w_alog.unsafe_ptr()
    w_alog_p[0] = 0.0
    w_alog_p[1] = 0.5
    var w_dtbias = make_list(nvh, 0.0)
    var wdt_p = w_dtbias.unsafe_ptr()
    wdt_p[0] = 0.0
    wdt_p[1] = 0.1

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

    var w_conv_bias_q = List[Float32]()
    var w_conv_bias_k = List[Float32]()
    var w_conv_bias_v = List[Float32]()

    var state = DeltaNetState(cfg)

    var hidden = make_list_seq(d, start=0.1, step=0.1)
    var result = make_list(d, 0.0)

    deltanet_forward(
        hidden,
        result,
        w_qkv,
        w_z,
        w_b,
        w_a,
        w_out,
        w_alog,
        w_dtbias,
        w_conv_q,
        w_conv_k,
        w_conv_v,
        w_conv_bias_q,
        w_conv_bias_k,
        w_conv_bias_v,
        state,
        cfg,
    )

    var any_nonzero = False
    for i in range(d):
        if result[i] != 0.0:
            any_nonzero = True
            break
    assert_true(
        any_nonzero, "deltanet_forward should produce a non-zero first output"
    )
    expect_close(result[0], 5.879743e-05, "deltanet_forward result[0]", 1e-6)
    expect_close(result[7], 0.0271918, "deltanet_forward result[7]", 1e-5)

    var state_updated = False
    for i in range(cfg.total_state_floats()):
        if state.s_matrix[i] != 0.0:
            state_updated = True
            break
    assert_true(state_updated, "deltanet_forward should update recurrent state")

    var result2 = make_list(d, 0.0)
    deltanet_forward(
        hidden,
        result2,
        w_qkv,
        w_z,
        w_b,
        w_a,
        w_out,
        w_alog,
        w_dtbias,
        w_conv_q,
        w_conv_k,
        w_conv_v,
        w_conv_bias_q,
        w_conv_bias_k,
        w_conv_bias_v,
        state,
        cfg,
    )

    var outputs_differ = False
    for i in range(d):
        if result[i] != result2[i]:
            outputs_differ = True
            break
    assert_true(
        outputs_differ,
        (
            "deltanet_forward should produce a different result after state"
            " updates"
        ),
    )
    print("  PASS forward_small")


# ===----------------------------------------------------------------------=== #
# O(1) Memory Property Test
# ===----------------------------------------------------------------------=== #


def test_o1_memory() raises:
    """Verify that DeltaNet state size is constant regardless of sequence length.
    """
    var cfg = DeltaNetConfig()
    var state_bytes = cfg.total_state_floats() * 4
    var expected = cfg.num_value_heads * cfg.key_dim * cfg.value_dim * 4
    assert_equal(
        state_bytes,
        expected,
        "DeltaNet state size should match the O(1) memory formula",
    )
    assert_equal(
        30 * state_bytes // 1024,
        61440,
        "30-layer DeltaNet state footprint changed unexpectedly",
    )
    print("  PASS o1_memory")


# ===----------------------------------------------------------------------=== #
# Main
# ===----------------------------------------------------------------------=== #


def run_suite() raises:
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
    print("All DeltaNet tests passed!")
    print("=" * 60)


def main() raises:
    run_suite()
