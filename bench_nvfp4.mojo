# NVFP4 Dequantization Benchmark
# Measures throughput of FP4 E2M1 dequantization

from time import perf_counter_ns
from collections import List
from random import random_ui64

from nvfp4.config import (
    NVFP4_GROUP_SIZE,
    FP4_E2M1_LUT,
    unpack_fp4_pair,
    dequant_fp4,
    dequant_fp4_simd,
)


fn benchmark_scalar_dequant(iterations: Int, num_elements: Int) -> Float64:
    """Benchmark scalar FP4 dequantization."""
    # Generate random packed FP4 values (each byte = 2 FP4 values)
    var packed_data = List[UInt8](capacity=num_elements // 2)
    var scales = List[Float32](capacity=num_elements // NVFP4_GROUP_SIZE)
    var output = List[Float32](capacity=num_elements)

    # Initialize with random data
    for i in range(num_elements // 2):
        packed_data.append(UInt8(random_ui64(0, 255)))
    for i in range(num_elements // NVFP4_GROUP_SIZE):
        scales.append(Float32(random_ui64(1, 100)) / 10.0)
    for i in range(num_elements):
        output.append(0.0)

    # Warmup
    for _ in range(10):
        for i in range(num_elements // 2):
            var unpacked = unpack_fp4_pair(packed_data[i])
            var scale_idx = (i * 2) // NVFP4_GROUP_SIZE
            var scale = scales[scale_idx]
            output[i * 2] = dequant_fp4(unpacked[0], scale)
            output[i * 2 + 1] = dequant_fp4(unpacked[1], scale)

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(iterations):
        for i in range(num_elements // 2):
            var unpacked = unpack_fp4_pair(packed_data[i])
            var scale_idx = (i * 2) // NVFP4_GROUP_SIZE
            var scale = scales[scale_idx]
            output[i * 2] = dequant_fp4(unpacked[0], scale)
            output[i * 2 + 1] = dequant_fp4(unpacked[1], scale)
    var end = perf_counter_ns()

    var total_elements = Float64(iterations * num_elements)
    var elapsed_ns = Float64(end - start)
    var throughput_gops = total_elements / elapsed_ns  # G elements/s
    return throughput_gops


fn benchmark_simd_dequant(iterations: Int, num_elements: Int) -> Float64:
    """Benchmark SIMD FP4 dequantization (16-wide)."""
    # Generate random packed FP4 values
    var packed_data = List[UInt8](capacity=num_elements // 2)
    var scales = List[Float32](capacity=num_elements // NVFP4_GROUP_SIZE)
    var output = List[Float32](capacity=num_elements)

    # Initialize with random data
    for i in range(num_elements // 2):
        packed_data.append(UInt8(random_ui64(0, 255)))
    for i in range(num_elements // NVFP4_GROUP_SIZE):
        scales.append(Float32(random_ui64(1, 100)) / 10.0)
    for i in range(num_elements):
        output.append(0.0)

    # Process in groups of 16 (matching NVFP4_GROUP_SIZE)
    var num_groups = num_elements // NVFP4_GROUP_SIZE

    # Warmup
    for _ in range(10):
        for g in range(num_groups):
            var base_idx = g * NVFP4_GROUP_SIZE
            var scale = scales[g]
            # Load 8 packed bytes (16 FP4 values)
            var fp4_vals = SIMD[DType.uint8, 16]()
            for i in range(8):
                var packed = packed_data[base_idx // 2 + i]
                fp4_vals[i * 2] = packed & 0x0F
                fp4_vals[i * 2 + 1] = (packed >> 4) & 0x0F
            # Dequantize
            var result = dequant_fp4_simd(fp4_vals, scale)
            # Store
            for i in range(16):
                output[base_idx + i] = result[i]

    # Benchmark
    var start = perf_counter_ns()
    for _ in range(iterations):
        for g in range(num_groups):
            var base_idx = g * NVFP4_GROUP_SIZE
            var scale = scales[g]
            var fp4_vals = SIMD[DType.uint8, 16]()
            for i in range(8):
                var packed = packed_data[base_idx // 2 + i]
                fp4_vals[i * 2] = packed & 0x0F
                fp4_vals[i * 2 + 1] = (packed >> 4) & 0x0F
            var result = dequant_fp4_simd(fp4_vals, scale)
            for i in range(16):
                output[base_idx + i] = result[i]
    var end = perf_counter_ns()

    var total_elements = Float64(iterations * num_elements)
    var elapsed_ns = Float64(end - start)
    var throughput_gops = total_elements / elapsed_ns
    return throughput_gops


fn main():
    print("=" * 60)
    print("NVFP4 Dequantization Benchmark (Mojo 0.26.1)")
    print("=" * 60)
    print()

    # Test sizes matching typical LLM layer dimensions
    var iterations = 1000

    print("Scalar Dequantization:")
    print("-" * 60)
    print("Elements    | Throughput (G elem/s) | Bandwidth (GB/s)")
    print("-" * 60)

    # Run benchmarks for different sizes
    var size1 = 1024
    var t1 = benchmark_scalar_dequant(iterations, size1)
    var b1 = t1 * 4.75
    print(size1, "      |", t1, "          |", b1)

    var size2 = 4096
    var t2 = benchmark_scalar_dequant(iterations, size2)
    var b2 = t2 * 4.75
    print(size2, "      |", t2, "          |", b2)

    var size3 = 16384
    var t3 = benchmark_scalar_dequant(iterations, size3)
    var b3 = t3 * 4.75
    print(size3, "     |", t3, "          |", b3)

    var size4 = 65536
    var t4 = benchmark_scalar_dequant(iterations, size4)
    var b4 = t4 * 4.75
    print(size4, "     |", t4, "          |", b4)

    var size5 = 262144
    var t5 = benchmark_scalar_dequant(iterations, size5)
    var b5 = t5 * 4.75
    print(size5, "    |", t5, "          |", b5)

    var size6 = 1048576
    var t6 = benchmark_scalar_dequant(iterations, size6)
    var b6 = t6 * 4.75
    print(size6, "   |", t6, "          |", b6)

    print()
    print("SIMD Dequantization (16-wide):")
    print("-" * 60)
    print("Elements    | Throughput (G elem/s) | Bandwidth (GB/s)")
    print("-" * 60)

    var st1 = benchmark_simd_dequant(iterations, size1)
    var sb1 = st1 * 4.75
    print(size1, "      |", st1, "          |", sb1)

    var st2 = benchmark_simd_dequant(iterations, size2)
    var sb2 = st2 * 4.75
    print(size2, "      |", st2, "          |", sb2)

    var st3 = benchmark_simd_dequant(iterations, size3)
    var sb3 = st3 * 4.75
    print(size3, "     |", st3, "          |", sb3)

    var st4 = benchmark_simd_dequant(iterations, size4)
    var sb4 = st4 * 4.75
    print(size4, "     |", st4, "          |", sb4)

    var st5 = benchmark_simd_dequant(iterations, size5)
    var sb5 = st5 * 4.75
    print(size5, "    |", st5, "          |", sb5)

    var st6 = benchmark_simd_dequant(iterations, size6)
    var sb6 = st6 * 4.75
    print(size6, "   |", st6, "          |", sb6)

    print()
    print("=" * 60)
    print("Note: FP4 E2M1 dequant = LUT lookup + scale multiply")
    print("Memory: 0.5B packed + 0.25B scale -> 4B float32 output")
    print("=" * 60)
