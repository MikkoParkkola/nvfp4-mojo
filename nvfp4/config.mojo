# ===----------------------------------------------------------------------=== #
# NVFP4 Configuration and Core Types for MAX
#
# NVFP4 (ModelOpt FP4) uses:
#   - E2M1 format: 2 exponent bits, 1 mantissa bit
#   - FP8 E4M3 blockscales for per-group scaling
#   - Group size of 16 (16 FP4 values share one blockscale)
#   - Packed as 2 FP4 values per uint8 byte
# ===----------------------------------------------------------------------=== #


# NVFP4 constants.
comptime NVFP4_GROUP_SIZE: Int = 16
comptime NVFP4_PACK_FACTOR: Int = 2  # 2 FP4 values per byte.

# FP4 E2M1 lookup table for dequantization.
# Maps 4-bit values (0-15) to their floating point representations.
comptime FP4_E2M1_LUT = SIMD[DType.float32, 16](
    0.0,    # 0000: +0
    0.5,    # 0001: +0.5
    1.0,    # 0010: +1.0
    1.5,    # 0011: +1.5
    2.0,    # 0100: +2.0
    3.0,    # 0101: +3.0
    4.0,    # 0110: +4.0
    6.0,    # 0111: +6.0
    -0.0,   # 1000: -0
    -0.5,   # 1001: -0.5
    -1.0,   # 1010: -1.0
    -1.5,   # 1011: -1.5
    -2.0,   # 1100: -2.0
    -3.0,   # 1101: -3.0
    -4.0,   # 1110: -4.0
    -6.0,   # 1111: -6.0
)


struct NvFp4Config(Copyable, Movable):
    """Configuration for NVFP4 quantization."""
    var group_size: Int
    var weight_dtype: DType
    var scale_dtype: DType
    var output_dtype: DType

    def __init__(out self):
        self.group_size = NVFP4_GROUP_SIZE
        self.weight_dtype = DType.uint8
        self.scale_dtype = DType.float8_e4m3fn
        self.output_dtype = DType.bfloat16

    def __init__(out self, *, copy: Self):
        self.group_size = copy.group_size
        self.weight_dtype = copy.weight_dtype
        self.scale_dtype = copy.scale_dtype
        self.output_dtype = copy.output_dtype

    def __init__(out self, *, deinit take: Self):
        self.group_size = take.group_size
        self.weight_dtype = take.weight_dtype
        self.scale_dtype = take.scale_dtype
        self.output_dtype = take.output_dtype


struct NvFp4Linear(Copyable, Movable):
    """NVFP4 quantized linear layer metadata.

    Note: Actual weight pointers are passed to forward functions
    separately to avoid complex lifetime tracking.
    """
    var in_features: Int
    var out_features: Int
    var input_scale: Float32
    var weight_scale_2: Float32  # Second-level scale from calibration.

    def __init__(out self, in_features: Int, out_features: Int):
        self.in_features = in_features
        self.out_features = out_features
        self.input_scale = 1.0
        self.weight_scale_2 = 1.0

    def __init__(out self, *, copy: Self):
        self.in_features = copy.in_features
        self.out_features = copy.out_features
        self.input_scale = copy.input_scale
        self.weight_scale_2 = copy.weight_scale_2

    def __init__(out self, *, deinit take: Self):
        self.in_features = take.in_features
        self.out_features = take.out_features
        self.input_scale = take.input_scale
        self.weight_scale_2 = take.weight_scale_2

    def alpha(self) -> Float32:
        """Compute combined scaling factor."""
        return self.input_scale * self.weight_scale_2

    def packed_weight_size(self) -> Int:
        """Size of packed weight buffer in bytes."""
        return self.out_features * (self.in_features // NVFP4_PACK_FACTOR)

    def scale_size(self) -> Int:
        """Size of blockscale buffer in elements."""
        return self.out_features * (self.in_features // NVFP4_GROUP_SIZE)


@always_inline
def unpack_fp4_pair(packed: UInt8) -> Tuple[UInt8, UInt8]:
    """Unpack a byte containing two FP4 values.

    Returns tuple of (lo_value, hi_value) as 4-bit indices.
    """
    var lo = packed & 0x0F
    var hi = (packed >> 4) & 0x0F
    return (lo, hi)


@always_inline
def dequant_fp4(fp4_val: UInt8, scale: Float32) -> Float32:
    """Dequantize a single FP4 value using lookup table and scale."""
    var base = FP4_E2M1_LUT[Int(fp4_val)]
    return base * scale


@always_inline
def dequant_fp4_simd[width: Int](
    fp4_vals: SIMD[DType.uint8, width],
    scales: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    """Vectorized FP4 dequantization."""
    var result = SIMD[DType.float32, width]()

    for i in range(width):
        result[i] = dequant_fp4(fp4_vals[i], scales[i])

    return result


def pack_fp4_pair(lo: UInt8, hi: UInt8) -> UInt8:
    """Pack two FP4 values into a single byte."""
    return (hi << 4) | (lo & 0x0F)


def compute_effective_scale(
    blockscale: Float32,
    weight_scale_2: Float32,
    input_scale: Float32,
) -> Float32:
    """Compute the effective scale for dequantization."""
    return blockscale * weight_scale_2 * input_scale
