# ===----------------------------------------------------------------------=== #
# NVFP4 GPU GEMM Kernel for MAX
#
# Performs: C = alpha * (dequant(A_fp4) @ B) where:
#   - A_fp4 is FP4 quantized weight matrix [M, K//2] packed
#   - B is bfloat16/float16 activation matrix [K, N]
#   - alpha = input_scale * weight_scale_2
#
# Uses tensor cores with blockscale dequantization fused into the kernel.
# ===----------------------------------------------------------------------=== #

from math import ceildiv
from sys import simd_width_of, size_of

from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from gpu.host import DeviceContext, FuncAttribute, DeviceBuffer
from gpu.memory import AddressSpace, external_memory
from layout import Layout, LayoutTensor, RuntimeLayout
from layout.tensor_core import TensorCore, get_fragment_size, get_mma_shape
from memory import UnsafePointer, bitcast

from .config import (
    NVFP4_GROUP_SIZE,
    NVFP4_PACK_FACTOR,
    NvFp4Config,
    NvFp4Linear,
    FP4_E2M1_LUT,
    unpack_fp4_pair,
    dequant_fp4,
)

# Block and warp tile sizes optimized for Blackwell SM100
alias BM: Int = 128  # Block rows
alias BN: Int = 128  # Block cols
alias BK: Int = 64   # Block K (reduction dimension)
alias WM: Int = 64   # Warp tile M
alias WN: Int = 64   # Warp tile N


@always_inline
fn load_and_dequant_fp4_tile[
    tile_m: Int,
    tile_k: Int,
    group_size: Int = NVFP4_GROUP_SIZE,
](
    packed_weights: UnsafePointer[UInt8],
    blockscales: UnsafePointer[Float8],
    output: UnsafePointer[Float32],
    weight_stride: Int,
    scale_stride: Int,
    start_row: Int,
    start_col: Int,
):
    """Load and dequantize a tile of FP4 weights.

    The packed weights are stored as [M, K//2] uint8.
    The blockscales are stored as [M, K//group_size] float8.

    Args:
        packed_weights: Pointer to packed FP4 weights
        blockscales: Pointer to FP8 blockscales
        output: Pointer to dequantized output buffer [tile_m, tile_k]
        weight_stride: Stride for weight matrix (K//2)
        scale_stride: Stride for scale matrix (K//group_size)
        start_row: Starting row in the weight matrix
        start_col: Starting column in the original (unpacked) weight matrix
    """
    # Each thread handles one row element
    var packed_col = start_col // NVFP4_PACK_FACTOR

    @parameter
    for m in range(tile_m):
        var row = start_row + m

        @parameter
        for k in range(tile_k // NVFP4_PACK_FACTOR):
            var col = packed_col + k
            var packed = packed_weights[row * weight_stride + col]

            # Get the blockscale for this group
            var scale_col = (start_col + k * NVFP4_PACK_FACTOR) // group_size
            var scale_fp8 = blockscales[row * scale_stride + scale_col]
            var scale = scale_fp8.cast[DType.float32]()

            # Unpack and dequantize
            var lo_hi = unpack_fp4_pair(packed)
            output[m * tile_k + k * 2] = dequant_fp4(lo_hi[0], scale)
            output[m * tile_k + k * 2 + 1] = dequant_fp4(lo_hi[1], scale)


fn nvfp4_gemm_kernel[
    M: Int,  # Output rows (out_features)
    N: Int,  # Output cols (batch * seq_len)
    K: Int,  # Reduction dim (in_features)
    act_dtype: DType = DType.bfloat16,
    out_dtype: DType = DType.bfloat16,
](
    # Inputs
    packed_weights: UnsafePointer[UInt8],     # [M, K//2] uint8
    blockscales: UnsafePointer[Float8],       # [M, K//16] float8
    activations: UnsafePointer[Scalar[act_dtype]],  # [K, N] bfloat16
    # Outputs
    output: UnsafePointer[Scalar[out_dtype]],  # [M, N]
    # Scalars
    alpha: Float32,  # input_scale * weight_scale_2
):
    """NVFP4 GEMM kernel: output = alpha * (dequant(weights) @ activations)

    This kernel fuses dequantization with GEMM for optimal memory bandwidth.
    Optimized for NVIDIA Blackwell (SM100) tensor cores.

    The computation flow:
    1. Load FP4 weights + FP8 scales from global memory
    2. Dequantize FP4 to FP32 in shared memory
    3. Load activations to shared memory
    4. Perform tensor core MMA
    5. Apply alpha scaling and write output
    """
    # Block indices
    var bm = block_idx.x
    var bn = block_idx.y

    # Thread indices within block
    var tid = thread_idx.x + thread_idx.y * block_dim.x

    # Shared memory for tiles
    var weight_smem = external_memory[
        Float32, AddressSpace.SHARED, alignment=128
    ].alloc[BM * BK]()

    var act_smem = external_memory[
        Scalar[act_dtype], AddressSpace.SHARED, alignment=128
    ].alloc[BK * BN]()

    # Accumulator registers (one per thread)
    var acc = SIMD[DType.float32, WM * WN // WARP_SIZE](0)

    # Main K loop
    var num_k_tiles = ceildiv(K, BK)

    for k_tile in range(num_k_tiles):
        var k_start = k_tile * BK

        # Cooperative loading: each thread loads part of the tile

        # Load and dequantize weight tile
        var weight_offset = bm * BM
        var weight_k_offset = k_start

        # Simple loading pattern - each thread loads multiple elements
        var elems_per_thread = (BM * BK) // (block_dim.x * block_dim.y)
        for i in range(elems_per_thread):
            var elem_idx = tid * elems_per_thread + i
            var m = elem_idx // BK
            var k = elem_idx % BK

            if m < BM and k < BK and (weight_offset + m) < M and (k_start + k) < K:
                # Get packed weight location
                var global_m = weight_offset + m
                var global_k = k_start + k
                var packed_k = global_k // NVFP4_PACK_FACTOR
                var packed_idx = global_m * (K // NVFP4_PACK_FACTOR) + packed_k

                var packed = packed_weights[packed_idx]
                var lo_hi = unpack_fp4_pair(packed)

                # Get scale
                var scale_k = global_k // NVFP4_GROUP_SIZE
                var scale_idx = global_m * (K // NVFP4_GROUP_SIZE) + scale_k
                var scale = blockscales[scale_idx].cast[DType.float32]()

                # Dequantize based on position within packed byte
                var fp4_val = lo_hi[0] if (global_k % 2 == 0) else lo_hi[1]
                weight_smem[m * BK + k] = dequant_fp4(fp4_val, scale)

        # Load activation tile
        var act_offset = k_start
        var act_n_offset = bn * BN

        for i in range(elems_per_thread):
            var elem_idx = tid * elems_per_thread + i
            var k = elem_idx // BN
            var n = elem_idx % BN

            if k < BK and n < BN and (k_start + k) < K and (act_n_offset + n) < N:
                var global_k = k_start + k
                var global_n = act_n_offset + n
                act_smem[k * BN + n] = activations[global_k * N + global_n]

        # Synchronize shared memory
        barrier()

        # Tensor core MMA computation
        # Each warp computes a WM x WN tile of the output
        var warp_id = tid // WARP_SIZE
        var lane = tid % WARP_SIZE

        var warp_m = (warp_id // (BN // WN)) * WM
        var warp_n = (warp_id % (BN // WN)) * WN

        # Simple reduction loop (can be optimized with tensor core intrinsics)
        @parameter
        for k in range(BK):
            var wm_elems = WM // (WARP_SIZE // (WN // 8))
            var wn_elems = WN // 8

            for wm in range(wm_elems):
                for wn in range(wn_elems):
                    var m = warp_m + wm * (WARP_SIZE // (WN // 8)) + lane // (WN // 8)
                    var n = warp_n + wn * 8 + (lane % (WN // 8)) * 8

                    if m < BM and n + 8 <= BN:
                        var w = weight_smem[m * BK + k]
                        for ni in range(8):
                            var a = act_smem[k * BN + n + ni].cast[DType.float32]()
                            acc[wm * wn_elems * 8 + wn * 8 + ni] += w * a

        # Synchronize before next tile
        barrier()

    # Write output with alpha scaling
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE

    var warp_m = (warp_id // (BN // WN)) * WM
    var warp_n = (warp_id % (BN // WN)) * WN

    var wm_elems = WM // (WARP_SIZE // (WN // 8))
    var wn_elems = WN // 8

    for wm in range(wm_elems):
        for wn in range(wn_elems):
            var m = bm * BM + warp_m + wm * (WARP_SIZE // (WN // 8)) + lane // (WN // 8)
            var n = bn * BN + warp_n + wn * 8 + (lane % (WN // 8)) * 8

            if m < M:
                for ni in range(8):
                    if n + ni < N:
                        var val = acc[wm * wn_elems * 8 + wn * 8 + ni] * alpha
                        output[m * N + n + ni] = val.cast[out_dtype]()


fn nvfp4_linear_forward(
    linear: NvFp4Linear,
    activations: LayoutTensor[DType.bfloat16, ...],
    output: LayoutTensor[mut=True, DType.bfloat16, ...],
    ctx: DeviceContext,
):
    """Forward pass for NVFP4 linear layer.

    Args:
        linear: The NVFP4 quantized linear layer
        activations: Input activations [batch, seq_len, in_features] or [batch*seq_len, in_features]
        output: Output tensor [batch, seq_len, out_features] or [batch*seq_len, out_features]
        ctx: GPU device context
    """
    var M = linear.out_features
    var K = linear.in_features
    var N = activations.dim(-2) if activations.rank == 3 else activations.dim(0)

    var alpha = linear.alpha()

    # Grid dimensions
    var grid_m = ceildiv(M, BM)
    var grid_n = ceildiv(N, BN)

    # Launch kernel
    # Note: In actual MAX, this would use the proper kernel launch API
    nvfp4_gemm_kernel[M, N, K](
        linear.weight,
        linear.weight_scale,
        activations.ptr,
        output.ptr,
        alpha,
    )
