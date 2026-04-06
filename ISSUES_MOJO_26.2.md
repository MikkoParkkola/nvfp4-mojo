# Mojo 26.2 Upgrade Issues for nvfp4-mojo

> These issues should be created as GitHub issues once the repo is published.

---

## Issue 1: [BREAKING] Migrate LegacyUnsafePointer to UnsafePointer in nvfp4_gemm_gpu.mojo

**Labels**: bug, mojo-upgrade
**Priority**: P0 (compile failure prevention)

### Summary
`nvfp4/nvfp4_gemm_gpu.mojo` line 25 uses `LegacyUnsafePointer` — deprecated in 26.2 nightly, future compile error.

### Acceptance Criteria
- [ ] Replace `LegacyUnsafePointer` with `UnsafePointer` pattern
- [ ] Verify compile on Mojo 26.2 nightly
- [ ] Run `bench_nvfp4.mojo` — no perf regression
- [ ] Run `tests/test_nvfp4.mojo` — all pass

### Evidence
- Affected: `nvfp4/nvfp4_gemm_gpu.mojo:25`
- Severity: ⚠️ Deprecation → future compile error
- Risk: Low (~30min mechanical migration)

---

## Issue 2: Migrate NVFP4 GEMM kernels to TileTensor API

**Labels**: enhancement, mojo-upgrade
**Priority**: P1 (forward-compatibility)

### Summary
Modular migrated all 8 SM100 kernel host functions to TileTensor in 26.2 nightly. Includes NVFP4-specific: "Push TileTensor from MOGG to kernel for grouped NVFP4 1D1D."

### Spike Finding (2026-02-19)
⚠️ TileTensor is an **internal Modular API** — not publicly available in Mojo SDK.
This is NOT a blocker for our kernels. We can use existing `gpu.compute.mma` and
standard GPU primitives. Monitor upstream for when TileTensor becomes public.

### Acceptance Criteria
- [ ] Study Modular's TileTensor migration patterns in upstream commits
- [ ] Monitor TileTensor API availability in public Mojo SDK
- [ ] When available: refactor `nvfp4_gemm_gpu.mojo` to use TileTensor API
- [ ] Benchmark before/after (expect neutral or improvement)
- [ ] Verify compatibility with DeepSeek-V3.1-NVFP4 model config
- [ ] Update `config.mojo` if TileTensor requires new config params
- [ ] All tests pass on 26.2 nightly

### Evidence
- Commits 2026-02-14: SM100 TileTensor migration + NVFP4 1D1D TileTensor
- DeepSeek-V3.1-NVFP4 now in Modular's CI eval pipeline
- Spike: TileTensor is internal API, NOT publicly available (not a blocker)
- Risk: Low (monitor, don't block on it)

---

## Issue 3: Add AMD CDNA GPU support and cross-vendor data center testing

**Labels**: enhancement, mojo-upgrade
**Priority**: P1 (cross-vendor validation)

### Summary
Mojo 26.2 supports AMD CDNA (MI300X/MI355X) as Tier 1 target alongside NVIDIA. NVFP4 kernels should be validated across both data-center GPU vendors.

### Spike Validation (2026-02-19)
- **NVIDIA CUDA**: ✅ PASS (Tier 1, production-ready)
- **AMD CDNA (MI300X/MI355X)**: ✅ PASS (Tier 1, data center)
- **AMD RDNA (consumer)**: ❌ KILL (Tier 3, no GenAI kernel support)
- **Apple Metal**: ❌ KILL (Tier 3, missing kernel pathways)
- FP4 E2M1 is part of OCP MX open standard (AMD+NVIDIA+Intel+Arm+Qualcomm)
- AMD CDNA has FP8 matrix cores — NVFP4 dequant to FP8 is feasible
- Cross-compile via `@parameter if is_nvidia_gpu()` / `is_amd_gpu()`

### Acceptance Criteria
- [ ] Test NVFP4 GEMM kernel compilation on AMD CDNA target (MI300X)
- [ ] Implement `@parameter if` branches for NVIDIA vs AMD CDNA paths
- [ ] Benchmark FP4→FP8 dequant on AMD CDNA matrix cores
- [ ] Verify kernel portability (same source → 2 data center GPU vendors)
- [ ] Update README with supported GPU targets (NVIDIA + AMD CDNA)

### Evidence
- Commit 2026-02-17: "Enable common MAX models to run on AMD RDNA GPUs"
- Spike: AMD CDNA is Tier 1 (same as NVIDIA), RDNA/Apple are Tier 3
- OCP MX standard includes FP4 E2M1 — hardware-agnostic format
- Risk: Low-Medium (CDNA is well-supported, kernel adjustments needed)
- Coverage: Most NVFP4 implementations are NVIDIA-only; this issue extends validation to both data-center vendors

---

## Issue 4: Adopt typed errors for GPU error handling

**Labels**: enhancement, mojo-upgrade
**Priority**: P2 (quality improvement)

### Summary
Mojo 26.1 typed errors provide zero-overhead error handling on GPUs. Apply to NVFP4 kernel error paths.

### Acceptance Criteria
- [ ] Define `NvFp4Error` typed error type
- [ ] Migrate error paths in `nvfp4_gemm_gpu.mojo` to typed errors
- [ ] Verify zero overhead via benchmark
- [ ] All tests pass

---

## Issue 5: Apply linear types for GPU resource lifecycle

**Labels**: enhancement, mojo-upgrade
**Priority**: P2 (safety improvement)

### Summary
Mojo 26.1 explicitly destroyed types ("linear types") provide compile-time guarantees that GPU resources cannot be leaked. Apply to NVFP4 buffer management.

### Acceptance Criteria
- [ ] Identify GPU buffers/resources in NVFP4 kernels
- [ ] Apply linear type annotations for compile-time leak prevention
- [ ] Verify compile-time enforcement catches intentional leaks in tests

---

## Issue 6: [BREAKING] Fix copyinit/moveinit parameter naming (other → copy/take)

**Labels**: bug, mojo-upgrade
**Priority**: P0 (compile failure prevention)

### Summary
Mojo 26.2 enforces `copy` for `__copyinit__` and `take` for `__moveinit__` source params. `config.mojo` uses `other` in 4 instances.

### Affected
- `nvfp4/config.mojo:51` — `__copyinit__(out self, other: Self)` → `copy: Self`
- `nvfp4/config.mojo:57` — `__moveinit__(out self, deinit other: Self)` → `deinit take: Self`
- `nvfp4/config.mojo:81` — `__copyinit__(out self, other: Self)` → `copy: Self`
- `nvfp4/config.mojo:87` — `__moveinit__(out self, deinit other: Self)` → `deinit take: Self`

### Acceptance Criteria
- [ ] Rename all `other` params to `copy`/`take` respectively
- [ ] Update all body references (`other.field` → `copy.field` / `take.field`)
- [ ] Verify compile on 26.2 nightly
- [ ] All tests pass

### Evidence
- Mojo 26.2: "Require copyinit/moveinit name consistency"
- Severity: ⚠️ Will become compile error
- Risk: Low (mechanical, 4 instances + body references)

---

## Issue 7: Adopt paged KV cache store helpers for inference optimization

**Labels**: enhancement, mojo-upgrade
**Priority**: P1 (critical for inference perf)

### Summary
Mojo 26.2 adds paged KV cache store helpers. Direct relevance to NVFP4 inference — paged KV cache is essential for efficient LLM serving with quantized models.

### Acceptance Criteria
- [ ] Study Modular's paged KV cache store helper API
- [ ] Evaluate integration with NVFP4 dequantization pipeline
- [ ] If beneficial: add KV cache management layer using official helpers
- [ ] Benchmark: inference throughput with/without paged KV cache
- [ ] Document integration with DeepSeek-V3.1-NVFP4 model

### Evidence
- Commit 2026-02-14: "[Kernels] Add paged KV cache store helpers"
- Risk: Medium (new API, but critical for production inference)

---

## Issue 8: Align with Modular's DeepSeek-V3.1-NVFP4 eval pipeline

**Labels**: enhancement
**Priority**: P1 (validation + alignment)

### Summary
Modular added DeepSeek-V3.1-NVFP4 to their official CI with LongBench-v2 evaluation. Our NVFP4 kernels should align with their eval methodology for quality assurance and upstream contribution potential.

### Acceptance Criteria
- [ ] Study Modular's DeepSeek-V3.1-NVFP4 eval config
- [ ] Run our NVFP4 kernels against same LongBench-v2 benchmark
- [ ] Compare quality metrics (logit accuracy, perplexity)
- [ ] Document any differences from upstream implementation
- [ ] Evaluate upstream contribution opportunity

### Evidence
- Commit 2026-02-17: "Add DeepSeek-V3.1-NVFP4 LongBench-v2 eval config"
- Commit 2026-02-17: "Use base MoE with per-expert weights for Qwen3-MoE"
- Strategic: Modular officially supports NVFP4 → our work is validated

---

## Issue 9: Test overlap scheduling gains for DeepSeek NVFP4

**Labels**: enhancement
**Priority**: P3 (free perf win)

### Summary
Mojo 26.2 automatically enables overlap scheduling for DeepSeek models. Verify this benefits our NVFP4 DeepSeek inference path.

### Acceptance Criteria
- [ ] Verify overlap scheduling auto-enables with our NVFP4 kernels
- [ ] Benchmark: throughput with/without overlap scheduling
- [ ] Document measured gains

### Evidence
- Commit 2026-02-16: "Automatically enable overlap scheduling for Deepseek"
- Risk: None (auto-enabled, just verify and measure)

---

## Issue 10: Re-evaluate NVFP4 quantization trade-offs as HBM supply expands (Micron $200B)

**Labels**: enhancement, strategic
**Priority**: P2 (forward planning)

### Source
[WSJ: Micron $200B AI Memory Investment](https://www.wsj.com/tech/micron-is-spending-200-billion-to-break-the-ai-memory-bottleneck-a4cc74a1) — Feb 2026

### Summary
Micron investing $200B to expand HBM manufacturing — biggest memory supply expansion in 40 years. Memory bandwidth is the primary bottleneck that drives quantization: NVFP4 exists because we need to fit models in limited HBM. As HBM supply increases, quantization trade-offs shift.

### Analysis Needed
- [ ] Track HBM capacity roadmap: HBM3E (24GB/stack) → HBM4 (48GB/stack) → HBM4E
- [ ] Model the crossover: at what HBM capacity does FP8 become "good enough" vs FP4?
- [ ] Benchmark NVFP4 vs FP8 quality gap at different model sizes
- [ ] Evaluate: does more HBM shift value prop from "fits in memory" to "faster inference"?
- [ ] Position nvfp4-mojo for BOTH scenarios: memory-constrained AND bandwidth-optimized

### Hypothesis
NVFP4 remains valuable even with abundant HBM because:
1. Model sizes grow faster than HBM capacity
2. FP4 gives 2× bandwidth efficiency → faster inference even when memory isn't scarce
3. Cost efficiency: less HBM needed per GPU = cheaper inference

### Evidence
- Micron $200B confirms HBM is THE bottleneck (validates nvfp4-mojo's premise)
- HBM4 timeline: 2026-2027 production
- Risk: Medium (whether HBM expansion outpaces model growth is uncertain)

---

## Issue 11: Implement DeltaNet inference kernel in Mojo

**Labels**: enhancement, architecture
**Priority**: P1 (botnaut core architecture)

### Summary
Gated DeltaNet (linear attention with delta rule) validated as ideal attention replacement for botnaut. All core operations confirmed implementable in Mojo GPU. DeltaNet provides addressable key-value state matrix (surgical memory updates) vs Mamba-2's compressed blob — critical for continual learning.

### Spike Validation (2026-02-19)
- **GEMM**: ✅ Already exists in `nvfp4_gemm_gpu.mojo`
- **Outer product (key × value)**: ✅ Trivial to implement
- **Element-wise gating**: ✅ Standard element-wise ops
- **Recurrent state update**: ✅ Standard GPU pattern (sequential loop in kernel)
- **Chunkwise parallel scan**: ⚠️ PARTIAL — building blocks exist (`gpu.primitives.block.prefix_sum`, warp shuffle, mbarrier sync), matrix-valued scan needs custom impl (~1-2 weeks)
- **Mojo GPU primitives available**: `gpu.compute.mma`, `gpu.primitives.block.*`, `gpu.primitives.warp.*`, `gpu.sync.*`

### Acceptance Criteria
- [ ] Implement DeltaNet recurrent state update kernel (inference path)
- [ ] Implement gated delta rule: S_t = (α ⊙ S_{t-1}) + (β ⊙ (v_t ⊗ k_t))
- [ ] Fuse with NVFP4 dequantization (FP4 weights → DeltaNet state update)
- [ ] Benchmark vs standard attention on Spark/DGX
- [ ] Test with sequence lengths >4K tokens
- [ ] Validate O(1) memory per step (no growing KV cache)

### Architecture
```
DeltaNet State Update (per layer):
  S_new = gate * S_old + beta * outer(value, key)
  output = S_new @ query

Where S is d_k × d_v matrix (addressable key-value state)
DeltaNet ≈ online gradient descent at layer level (von Oswald 2022)
```

### Evidence
- DeltaNet paper: Yang et al. 2024 "Gated Delta Networks"
- Mathematically equivalent to TTT (Test-Time Training)
- All Mojo GPU primitives validated in spike
- Risk: Low for inference (~1 week), Medium for chunkwise prefill (~2 weeks)

---

## Issue 12: AMD CDNA NVFP4 pilot (1-2 day spike)

**Labels**: enhancement, pilot
**Priority**: P1 (cross-vendor validation)

### Summary
Validate NVFP4 FP4→FP8 dequantization on AMD CDNA (MI300X) matrix cores. FP4 E2M1 is part of OCP MX open standard — hardware-agnostic format means same quantized weights work on both NVIDIA and AMD.

### Spike Validation (2026-02-19)
- AMD CDNA is Tier 1 in Mojo (same support level as NVIDIA)
- AMD CDNA has FP8 matrix cores — dequant FP4→FP8 feeds directly into MFMA
- OCP MX standard ensures format portability
- Cross-compile: `@parameter if is_nvidia_gpu()` / `is_amd_gpu()` branching

### Acceptance Criteria
- [ ] Compile `nvfp4_gemm_gpu.mojo` targeting AMD CDNA (MI300X)
- [ ] Add `@parameter if` branches for NVIDIA vs AMD code paths
- [ ] Test FP4→FP8 dequant on AMD CDNA matrix cores (MFMA)
- [ ] Benchmark GEMM throughput: NVIDIA H100 vs AMD MI300X
- [ ] Document any AMD-specific kernel adjustments needed
- [ ] Verify same NVFP4 quantized weights work on both targets

### Evidence
- Spike: AMD CDNA is Tier 1, RDNA/Apple are Tier 3 (killed)
- OCP MX standard: FP4 E2M1 is vendor-neutral
- Risk: Low-Medium (Tier 1 support, but untested path)
- Timeline: 1-2 day spike on Spark infrastructure

---

## Issue 13: Qwen 3.5 NVFP4 weight loading (safetensors/GGUF)

**Labels**: enhancement, model-support
**Priority**: P1 (target model for botnaut)

### Summary
Qwen 3.5 (397B/17B active MoE) validated as target model architecture for botnaut. MXFP4_MOE GGUF variant already exists (unsloth). Safetensors format is framework-independent (no PyTorch needed).

### Spike Validation (2026-02-19)
- **Safetensors format**: ✅ 8-byte header + JSON metadata + raw tensor bytes
- **GGUF MXFP4_MOE variant**: ✅ Exists (unsloth, 6 shards)
- **Vision encoder**: ✅ Separate `mmproj-BF16.gguf` (918MB)
- **No PyTorch dependency**: ✅ Both formats are framework-independent

### Acceptance Criteria
- [ ] Implement safetensors reader in Mojo (8-byte header + JSON + raw tensors)
- [ ] Implement GGUF reader for MXFP4_MOE quantized weights
- [ ] Load Qwen 3.5 MoE expert weights into NVFP4 kernel pipeline
- [ ] Verify weight format: 512 experts, 10 routed + 1 shared per layer
- [ ] Benchmark: weight loading time for full model (6 GGUF shards)
- [ ] Test with DeepSeek-V3.1-NVFP4 config as secondary target

### Evidence
- Qwen 3.5: 397B total, 17B active, Apache 2.0 license
- MXFP4 (OCP MX FP4) ≈ NVFP4 (E2M1) — same format, different branding
- Vision encoder separable — can add multimodal later
- Risk: Low (formats well-documented, GGUF widely adopted)

---

## Issue 14: Evaluate COMPOT orthogonal compression for NVFP4 pipeline

**Labels**: research, enhancement
**Priority**: P2 (potential quality improvement)

### Source
[COMPOT: Calibration-Optimized Matrix Procrustes Orthogonalization for Transformers Compression](https://arxiv.org/abs/2602.15200) — Makhov et al., Feb 2026

### Summary
COMPOT introduces training-free transformer compression using sparse dictionary learning with orthogonal dictionaries. Key properties:
- **Closed-form solutions** — no iterative optimization (fast, deterministic)
- **Adaptive per-layer compression** — redistributes compression budget based on layer sensitivity
- **Compatible with post-training quantization** — can layer on top of FP4/FP8 quantization
- Uses small calibration dataset (not full retraining)

### Relevance to NVFP4
Our current pipeline: FP16/BF16 → FP4 E2M1 quantization (blockscale FP8 E4M3, group_size=16). COMPOT could:
1. **Pre-compress** weights via orthogonal projection BEFORE FP4 quantization → better quality at same bit-width
2. **Adaptive layer budgets** — some layers tolerate more aggressive quantization than others. COMPOT's per-layer redistribution could inform which layers get FP4 vs FP8
3. **Complement NVFP4** — not a replacement, but an additional compression stage in the pipeline

### Acceptance Criteria
- [ ] Read full paper and extract algorithm details
- [ ] Evaluate: does orthogonal pre-compression improve FP4 quantization quality?
- [ ] Compare: COMPOT+FP4 vs plain FP4 on perplexity benchmarks
- [ ] If promising: implement orthogonal projection in Mojo as preprocessing step
- [ ] Assess computational cost of calibration step

### Evidence
- Paper claims improved performance vs existing low-rank and sparse alternatives
- Compatible with PTQ (post-training quantization) — our use case exactly
- Risk: Low (research evaluation, no code changes until validated)
- Timeline: 2-4 hour paper analysis + optional 1-day prototype
