# NVFP4 Test Runner
# Tests the core dequantization logic

from nvfp4.config import (
    NVFP4_GROUP_SIZE,
    NVFP4_PACK_FACTOR,
    FP4_E2M1_LUT,
    unpack_fp4_pair,
    dequant_fp4,
    pack_fp4_pair,
    NvFp4Config,
)


fn test_fp4_lut():
    """Test FP4 E2M1 lookup table values."""
    print("Testing FP4 E2M1 LUT:")
    for i in range(16):
        print("  [", i, "] =", FP4_E2M1_LUT[i])


fn test_pack_unpack():
    """Test packing and unpacking FP4 pairs."""
    print("\nTesting pack/unpack:")

    # Test: pack(3, 5) should give 0x53
    var packed = pack_fp4_pair(3, 5)
    print("  pack(3, 5) =", packed, "(expected 83 = 0x53)")

    # Unpack and verify
    var unpacked = unpack_fp4_pair(packed)
    print("  unpack(", packed, ") = (", unpacked[0], ",", unpacked[1], ")")


fn test_dequant():
    """Test dequantization with known values."""
    print("\nTesting dequantization:")

    # Test with scale = 1.0
    var scale: Float32 = 1.0
    print("  Scale = 1.0:")
    print("    dequant(0) =", dequant_fp4(0, scale), "(expected 0.0)")
    print("    dequant(1) =", dequant_fp4(1, scale), "(expected 0.5)")
    print("    dequant(2) =", dequant_fp4(2, scale), "(expected 1.0)")
    print("    dequant(7) =", dequant_fp4(7, scale), "(expected 6.0)")
    print("    dequant(9) =", dequant_fp4(9, scale), "(expected -0.5)")

    # Test with scale = 2.0
    scale = 2.0
    print("  Scale = 2.0:")
    print("    dequant(2) =", dequant_fp4(2, scale), "(expected 2.0)")
    print("    dequant(7) =", dequant_fp4(7, scale), "(expected 12.0)")


fn test_config():
    """Test NvFp4Config initialization."""
    print("\nTesting NvFp4Config:")
    var config = NvFp4Config()
    print("  group_size:", config.group_size, "(expected", NVFP4_GROUP_SIZE, ")")
    print("  weight_dtype:", config.weight_dtype)
    print("  scale_dtype:", config.scale_dtype)
    print("  output_dtype:", config.output_dtype)


fn main():
    print("=" * 50)
    print("NVFP4 Mojo Kernel Test Suite")
    print("=" * 50)

    test_fp4_lut()
    test_pack_unpack()
    test_dequant()
    test_config()

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
