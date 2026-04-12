# NVFP4 Mojo Kernels

NVFP4 (ModelOpt FP4) quantization kernels implemented in Mojo for MAX inference engine.

## Overview

NVFP4 uses E2M1 format (2 exponent bits, 1 mantissa bit) with FP8 E4M3 blockscales:
- **Values**: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
- **Group size**: 16 (16 FP4 values share one FP8 blockscale)
- **Packing**: 2 FP4 values per byte (uint8)

## Files

```
nvfp4-mojo/
├── nvfp4/                    # Main package
│   ├── __init__.mojo         # Package exports
│   ├── config.mojo           # Configuration, constants, dequantization
│   ├── deltanet.mojo         # DeltaNet support code
│   ├── nvfp4_gemm_gpu.mojo   # GPU GEMM kernel with fused dequantization
│   └── weight_loader.mojo    # Weight loading from safetensors
├── tests/
│   ├── test_deltanet.mojo    # DeltaNet-focused tests
│   └── test_nvfp4.mojo       # Core NVFP4 tests
├── run_tests.mojo            # Test runner
├── mojoproject.toml          # Project configuration
└── README.md
```

## Running Tests

```bash
pixi run mojo run run_tests.mojo
```

The project runner executes both suites and exits non-zero on the first
assertion failure. To iterate on one suite at a time:

```bash
pixi run mojo run -I . tests/test_nvfp4.mojo
pixi run mojo run -I . tests/test_deltanet.mojo
```

## Weight Structure

NVFP4 checkpoints contain:
```
model.layers.X.mlp.experts.Y.gate_up_proj.weight          # [intermediate, hidden//2] uint8
model.layers.X.mlp.experts.Y.gate_up_proj.weight_scale    # [intermediate, hidden//16] float8_e4m3
model.layers.X.mlp.experts.Y.gate_up_proj.weight_scale_2  # [intermediate] float32
model.layers.X.mlp.experts.Y.gate_up_proj.input_scale     # [] float32
```

## Usage with SGLang (Working)

For DGX Spark (SM121), use the patched sgl_kernel:
```bash
# Install pre-built wheel with SM121 support
pip install /path/to/sgl_kernel-0.3.21-sm121-cp310-abi3-linux_aarch64.whl --force-reinstall

# Start server
python -m sglang.launch_server \
  --model-path /models/NVFP4-Qwen3-30B-A3B-Instruct-2507-FP4 \
  --quantization modelopt_fp4 \
  --mem-fraction-static 0.8 \
  --disable-cuda-graph
```

## Benchmarks (DGX Spark GB10)

| Test | Throughput |
|------|------------|
| Single request (128 tok) | 36 tok/s |
| Single request (256 tok) | 37 tok/s |
| Batch of 4 concurrent | 110 tok/s |

## Status

- ✅ SGLang integration working with SM121 patch (PR #17421)
- 🔄 MAX integration pending (requires MAX source build)
- ✅ Mojo kernel implementations complete
