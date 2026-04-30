# Mojo 1.0 Nightly Upgrade Evidence

Date: 2026-04-28

## Current compiler

```text
Mojo 1.0.0b1.dev2026042717 (4404aa67)
```

The Pixi lock now resolves Mojo to `1.0.0b1.dev2026042717` on the active platforms.

## Why this upgrade matters

Official Modular notes show Mojo moving through the 1.0 beta nightly line and MAX adding relevant GPU/runtime work. The MAX nightly changelog calls out improved NVFP4 grouped matmul performance on B200, plus broader eager interpreter and distributed execution work:

- https://docs.modular.com/mojo/changelog/
- https://docs.modular.com/max/nightly-changelog/
- https://forum.modular.com/t/max-nightly-26-3-0-dev2026042405-mojo-1-0-0b1-dev2026042405-released/2997

That makes the upgrade strategically relevant, but it does not remove the need for local performance gates.

## Code migration completed

- Replaced legacy `fn` declarations with `def`.
- Updated copy/move constructors to the 1.0 initializer spelling.
- Made stdlib imports explicit.
- Added pointer origin parameters for `UnsafePointer` call sites in GPU and file-loading paths.
- Updated string byte handling in the safetensors/GGUF loader to avoid Unicode-length ambiguity.
- Fixed the benchmark SIMD scale argument so it measures the intended 16-wide dequant path.

## Validation commands

All commands below pass with the 1.0 nightly compiler:

```bash
pixi run mojo run --Werror run_tests.mojo
pixi run mojo package --Werror nvfp4 -o /tmp/nvfp4.mojopkg
pixi run mojo run --Werror -I . bench_nvfp4.mojo
```

## CPU benchmark signal

Same benchmark, same machine, before and after the compiler upgrade.

| Path | 0.26.1 pin | 1.0 nightly | Signal |
| --- | ---: | ---: | --- |
| Scalar dequant, 1,048,576 elems | 1.242 G elem/s | 0.422 G elem/s | ~66% slower |
| SIMD dequant, 1,048,576 elems | 0.184 G elem/s | 0.182 G elem/s | roughly flat |

The 1.0 nightly run also showed scalar throughput around `0.39-0.44 G elem/s` across tested sizes and 16-wide SIMD around `0.177-0.187 G elem/s`.

## Recommendation

Keep the 1.0 migration branch current, but do not lock Botnaut-server fully onto Mojo 1.0 for performance-sensitive paths until GPU evidence exists.

Required gate before broader adoption:

1. Compile the Botnaut-server Mojo/Spark path against this 1.0 nightly.
2. Run NVFP4 grouped matmul and end-to-end decode benchmarks on the target GPU class, not just CPU dequant microbenchmarks.
3. Compare latency, throughput, and memory bandwidth against the current Rust/CUDA or existing Mojo baseline.
4. Accept the upgrade only if the intended GPU/Spark path is neutral or faster, or if a slower result is isolated to non-production CPU fallback code.

