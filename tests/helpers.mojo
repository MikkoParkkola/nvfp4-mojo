from std.testing import assert_true


def approx_eq(a: Float32, b: Float32, tol: Float32 = 1e-4) -> Bool:
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


def expect_close(
    actual: Float32,
    expected: Float32,
    label: String,
    tol: Float32 = 1e-4,
) raises:
    assert_true(
        approx_eq(actual, expected, tol),
        label + ": expected " + String(expected) + ", got " + String(actual),
    )
