# Test NVFP4 dequantization.
from std.testing import assert_equal

from nvfp4 import (
    NVFP4_GROUP_SIZE,
    NVFP4_PACK_FACTOR,
    FP4_E2M1_LUT,
    unpack_fp4_pair,
    dequant_fp4,
    pack_fp4_pair,
    NvFp4Config,
)
from tests.helpers import expect_close


def test_fp4_lut() raises:
    """Test FP4 E2M1 lookup table values."""
    expect_close(FP4_E2M1_LUT[0], 0.0, "FP4_E2M1_LUT[0]")
    expect_close(FP4_E2M1_LUT[1], 0.5, "FP4_E2M1_LUT[1]")
    expect_close(FP4_E2M1_LUT[2], 1.0, "FP4_E2M1_LUT[2]")
    expect_close(FP4_E2M1_LUT[3], 1.5, "FP4_E2M1_LUT[3]")
    expect_close(FP4_E2M1_LUT[4], 2.0, "FP4_E2M1_LUT[4]")
    expect_close(FP4_E2M1_LUT[5], 3.0, "FP4_E2M1_LUT[5]")
    expect_close(FP4_E2M1_LUT[6], 4.0, "FP4_E2M1_LUT[6]")
    expect_close(FP4_E2M1_LUT[7], 6.0, "FP4_E2M1_LUT[7]")
    expect_close(FP4_E2M1_LUT[8], -0.0, "FP4_E2M1_LUT[8]")
    expect_close(FP4_E2M1_LUT[9], -0.5, "FP4_E2M1_LUT[9]")
    expect_close(FP4_E2M1_LUT[10], -1.0, "FP4_E2M1_LUT[10]")
    expect_close(FP4_E2M1_LUT[11], -1.5, "FP4_E2M1_LUT[11]")
    expect_close(FP4_E2M1_LUT[12], -2.0, "FP4_E2M1_LUT[12]")
    expect_close(FP4_E2M1_LUT[13], -3.0, "FP4_E2M1_LUT[13]")
    expect_close(FP4_E2M1_LUT[14], -4.0, "FP4_E2M1_LUT[14]")
    expect_close(FP4_E2M1_LUT[15], -6.0, "FP4_E2M1_LUT[15]")
    print("  PASS fp4_lut")


def test_pack_unpack() raises:
    """Test packing and unpacking FP4 pairs."""
    var packed = pack_fp4_pair(UInt8(3), UInt8(5))
    assert_equal(
        packed, UInt8(83), "pack_fp4_pair should preserve nibble order"
    )

    var unpacked = unpack_fp4_pair(packed)
    assert_equal(unpacked[0], UInt8(3), "unpack_fp4_pair low nibble mismatch")
    assert_equal(unpacked[1], UInt8(5), "unpack_fp4_pair high nibble mismatch")
    print("  PASS pack_unpack")


def test_dequant() raises:
    """Test dequantization with known values."""
    var scale: Float32 = 1.0
    expect_close(dequant_fp4(UInt8(0), scale), 0.0, "dequant_fp4(0, 1.0)")
    expect_close(dequant_fp4(UInt8(1), scale), 0.5, "dequant_fp4(1, 1.0)")
    expect_close(dequant_fp4(UInt8(2), scale), 1.0, "dequant_fp4(2, 1.0)")
    expect_close(dequant_fp4(UInt8(7), scale), 6.0, "dequant_fp4(7, 1.0)")
    expect_close(dequant_fp4(UInt8(9), scale), -0.5, "dequant_fp4(9, 1.0)")

    scale = 2.0
    expect_close(dequant_fp4(UInt8(2), scale), 2.0, "dequant_fp4(2, 2.0)")
    expect_close(dequant_fp4(UInt8(7), scale), 12.0, "dequant_fp4(7, 2.0)")
    print("  PASS dequant")


def test_config() raises:
    """Test NvFp4Config initialization."""
    var config = NvFp4Config()
    assert_equal(NVFP4_PACK_FACTOR, 2, "NVFP4 pack factor should stay at 2")
    assert_equal(
        config.group_size,
        NVFP4_GROUP_SIZE,
        "NvFp4Config should default to the canonical group size",
    )
    assert_equal(
        config.weight_dtype, DType.uint8, "weight dtype should be uint8"
    )
    assert_equal(
        config.scale_dtype,
        DType.float8_e4m3fn,
        "scale dtype should be float8_e4m3fn",
    )
    assert_equal(
        config.output_dtype,
        DType.bfloat16,
        "output dtype should be bfloat16",
    )
    print("  PASS config")


def run_suite() raises:
    print("=" * 50)
    print("NVFP4 Mojo Kernel Test Suite")
    print("=" * 50)

    test_fp4_lut()
    test_pack_unpack()
    test_dequant()
    test_config()

    print("\n" + "=" * 50)
    print("All NVFP4 tests passed!")
    print("=" * 50)


def main() raises:
    run_suite()
