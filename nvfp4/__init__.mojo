# NVFP4 Mojo Package - Core GEMM kernel for Blackwell NVFP4 inference.
# Re-export public symbols from core modules.

from .config import (
    NVFP4_GROUP_SIZE,
    NVFP4_PACK_FACTOR,
    FP4_E2M1_LUT,
    NvFp4Config,
    NvFp4Linear,
    unpack_fp4_pair,
    dequant_fp4,
    dequant_fp4_simd,
    pack_fp4_pair,
    compute_effective_scale,
)

# Core GEMM kernel - the heart of NVFP4 inference.
from .nvfp4_gemm_gpu import (
    nvfp4_gemm_kernel,
    nvfp4_linear_forward,
    load_and_dequant_fp4_tile,
    BM,
    BN,
    BK,
    WM,
    WN,
    FP8_SCALE_DTYPE,
)

# DeltaNet inference kernel - gated linear attention with delta rule.
from .deltanet import (
    DeltaNetConfig,
    DeltaNetState,
    deltanet_forward,
    nvfp4_matvec,
    DN_NUM_KEY_HEADS,
    DN_NUM_VALUE_HEADS,
    DN_KEY_DIM,
    DN_VALUE_DIM,
    DN_V_GROUPS,
    DN_CONV_KERNEL,
    DN_RMS_EPS,
)

# Weight loading from safetensors and GGUF formats.
from .weight_loader import (
    TensorInfo,
    SafetensorsFile,
    LoadedWeight,
    Qwen35LayerWeights,
    GGUFTensorInfo,
    GGUFFile,
    open_safetensors,
    open_gguf,
    load_linear_safetensors,
    load_qwen35_layer,
    discover_safetensor_shards,
    open_model_shards,
)
