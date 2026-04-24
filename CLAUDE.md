# nvfp4-mojo — agent onboarding + project instructions

> **This file is the source of truth for agents working on nvfp4-mojo.** `AGENTS.md` is a symlink to this file.

**Status**: early · Mojo 26.2 · Apache-2.0 · public · strategic moat (see MIK-2975)

## Product Vision

nvfp4-mojo ships **NVFP4 (ModelOpt FP4) quantization kernels in Mojo** for the MAX inference engine. NVFP4 uses E2M1 format (2 exponent bits, 1 mantissa bit) with FP8 E4M3 blockscales, group size 16, packed 2 FP4 values per byte. This is the quantization substrate that unlocks sub-4-bit inference on DGX Spark (SM121 / GB10) and the long-horizon path to analog memristor hardware.

**Strategic role**: nvfp4-mojo is the quantization kernel in the Botnaut stack. It pairs with `bnaut-memory` / hebb for the full memory-substrate story documented in MIK-2975 (Hebbian × hebb × NVFP4 SUPERCHARGE). Keep the kernel surface clean and kernel-only; do not embed distribution or routing logic.

## Current Status

- **SGLang integration**: working with SM121 patch (sgl_kernel PR #17421)
- **MAX integration**: pending (requires MAX source build)
- **Benchmarks (DGX Spark GB10)**: single-request 128 tok → 36 tok/s · 256 tok → 37 tok/s · batch-of-4 concurrent → 110 tok/s
- **Toolchain**: Mojo 26.2 (post UnsafePointer + comptime migration, #17/#1/#6)
- **Build**: pixi-based, linux-aarch64 environment lock
- **License**: Apache-2.0 (unblocks external contributions)

## Plan Forward (near-term, technical)

- **MAX integration** — currently pending MAX source build; highest-value unlock
- **DeltaNet inference kernel + weight loader** landed (#11, #13); continue integration with botnaut-server
- **Mojo 26.2 migration** completed; keep lockstep with Modular releases
- **Strategic path (MIK-2975)**: GPU NVFP4 Hebbian-gate spike → analog memristor crossbar roadmap (3-year EXCEED target: 100× throughput-per-watt vs GPU FP16 attention)
- **COMPOT orthogonal compression evaluation** — issue #14 open

## Decisions Locked (do not re-litigate)

| Decision | Rationale | Do not |
|---|---|---|
| **E2M1 FP4 format + FP8 E4M3 blockscales, group size 16** | OCP MX standard; interoperable with NVIDIA ModelOpt | Change block size / mantissa bits without upstream coordination |
| **Packed uint8 storage: 2 FP4 per byte** | Memory density; aligns with ModelOpt checkpoint layout | Use sparse / unpacked storage in hot paths |
| **Fused dequantization in GEMM kernel** | Throughput; avoids materializing FP16 activations | Split dequantize and matmul into separate passes on hot paths |
| **Mojo as the implementation language** | Compile-time specialization + GPU codegen; MAX target | Port the kernel to Rust / CUDA C when Mojo is fit for purpose |
| **SGLang first-target runtime** (SM121 patch) | Spark validated; fastest path to real throughput numbers | Tie the kernel exclusively to one runtime; MAX is second target |
| **Benchmarks on real hardware (DGX Spark GB10 / SM121)** | Published numbers must be reproducible | Publish micro-benchmarks without naming the hardware target |
| **Apache-2.0 license** | External contributions; ModelOpt-adjacent ecosystem norm | Relicense without explicit user direction |
| **Scope: kernel-only** — no distribution, routing, serving | Keeps the surface testable and portable | Embed dispatch / routing / deployment into the kernel repo |

## Anti-Patterns (things agents get wrong in this repo)

- **Materializing FP16 activations in the GEMM hot path** — kills the memory-bandwidth advantage. Fused dequantization stays fused.
- **Coupling the kernel to botnaut-server directly** — the kernel must remain reusable by SGLang, MAX, and downstream integrators. No server-specific imports.
- **Changing block size / mantissa bits** without an OCP MX / ModelOpt coordination — interop is the moat.
- **Shipping benchmarks without hardware name** — "36 tok/s" means nothing without `DGX Spark GB10 / SM121`. Always name the platform.
- **Bypassing `pixi run`** — environment reproducibility depends on pixi lock for linux-aarch64.
- **Blocking external contributions with MIT-only or proprietary clauses** — Apache-2.0 is deliberate.

## Guidance for Agents

- **Before editing a kernel**: run `pixi run mojo run run_tests.mojo`; assertion failure means stop and diagnose.
- **Single-suite iteration**: `pixi run mojo run -I . tests/test_nvfp4.mojo` or `tests/test_deltanet.mojo`.
- **When adding a benchmark**: include hardware name (GB10 / SM121), tokenizer, model, batch size, measured metric. The public numbers above follow this format.
- **Strategic context**: MIK-2975 tracks the Hebbian-gate × NVFP4 SUPERCHARGE. If a change touches the Hebbian outer-product path or memristor-friendly formats, reference MIK-2975 in the commit.
- **Mojo migration discipline**: stay on Mojo 26.2+; prior migration commits (#17, #1, #6) document the UnsafePointer + comptime patterns.

## Where to Look

| You want to… | Read |
|---|---|
| Understand NVFP4 format | `nvfp4/config.mojo` |
| GPU GEMM kernel | `nvfp4/nvfp4_gemm_gpu.mojo` |
| Weight loading (safetensors → NVFP4) | `nvfp4/weight_loader.mojo` |
| DeltaNet support | `nvfp4/deltanet.mojo` |
| Tests | `tests/test_nvfp4.mojo` + `tests/test_deltanet.mojo` |
| SGLang integration | README §"Usage with SGLang (Working)" |
| Benchmarks | README §"Benchmarks (DGX Spark GB10)" |
| Strategic context | Linear MIK-2975 |

## AGENTS.md compatibility

`AGENTS.md` is a symlink to this file. Tools looking for either name land on the same content. Do not edit `AGENTS.md` directly; edit `CLAUDE.md`.
