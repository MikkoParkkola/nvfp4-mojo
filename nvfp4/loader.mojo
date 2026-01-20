# ===----------------------------------------------------------------------=== #
# NVFP4 Weight Loader for MAX
#
# Loads NVFP4 quantized weights from Hugging Face safetensors format.
# Handles the ModelOpt quantization config and weight structure.
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer
from collections import Dict
from pathlib import Path

from .config import (
    NVFP4_GROUP_SIZE,
    NvFp4Config,
    NvFp4Linear,
)
from .nvfp4_moe import NvFp4Expert, NvFp4MoELayer


struct NvFp4QuantConfig:
    """Parsed quantization config from hf_quant_config.json"""
    var quant_algo: String           # "NVFP4"
    var kv_cache_quant_algo: String  # "FP8"
    var group_size: Int              # 16
    var exclude_modules: List[String]

    fn __init__(out self):
        self.quant_algo = "NVFP4"
        self.kv_cache_quant_algo = "FP8"
        self.group_size = NVFP4_GROUP_SIZE
        self.exclude_modules = List[String]()


fn parse_quant_config(config_json: String) -> NvFp4QuantConfig:
    """Parse the hf_quant_config.json file.

    Expected format:
    {
        "producer": {"name": "modelopt", "version": "0.33.1"},
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": "FP8",
            "group_size": 16,
            "exclude_modules": ["model.layers.*.mlp.gate", "lm_head"]
        }
    }
    """
    var config = NvFp4QuantConfig()
    # TODO: Implement JSON parsing in Mojo
    # For now, use default values from the model we analyzed
    return config


struct SafetensorFile:
    """Represents a safetensors file with tensor metadata."""
    var path: String
    var tensor_names: List[String]

    fn __init__(out self, path: String):
        self.path = path
        self.tensor_names = List[String]()


fn get_tensor_shape(file: SafetensorFile, name: String) -> Tuple[Int, Int]:
    """Get the shape of a tensor from safetensors file."""
    # TODO: Implement safetensors header parsing
    return (0, 0)


fn load_tensor_uint8(
    file: SafetensorFile,
    name: String,
    out_ptr: UnsafePointer[UInt8],
    size: Int,
):
    """Load a uint8 tensor from safetensors file."""
    # TODO: Implement safetensors tensor loading
    pass


fn load_tensor_float8(
    file: SafetensorFile,
    name: String,
    out_ptr: UnsafePointer[Float8],
    size: Int,
):
    """Load a float8 tensor from safetensors file."""
    # TODO: Implement safetensors tensor loading
    pass


fn load_tensor_float32_scalar(
    file: SafetensorFile,
    name: String,
) -> Float32:
    """Load a scalar float32 from safetensors file."""
    # TODO: Implement safetensors tensor loading
    return 1.0


fn load_nvfp4_linear(
    file: SafetensorFile,
    prefix: String,  # e.g., "model.layers.0.mlp.experts.0.down_proj"
) -> NvFp4Linear:
    """Load an NVFP4 quantized linear layer.

    Expected tensors:
    - {prefix}.weight: [out, in//2] uint8
    - {prefix}.weight_scale: [out, in//16] float8
    - {prefix}.weight_scale_2: [] float32
    - {prefix}.input_scale: [] float32
    """
    var weight_name = prefix + ".weight"
    var scale_name = prefix + ".weight_scale"
    var scale2_name = prefix + ".weight_scale_2"
    var input_scale_name = prefix + ".input_scale"

    var shape = get_tensor_shape(file, weight_name)
    var out_features = shape[0]
    var in_features = shape[1] * 2  # Unpack from packed format

    var weight_size = out_features * (in_features // 2)
    var scale_size = out_features * (in_features // NVFP4_GROUP_SIZE)

    var weight = UnsafePointer[UInt8].alloc(weight_size)
    var weight_scale = UnsafePointer[Float8].alloc(scale_size)

    load_tensor_uint8(file, weight_name, weight, weight_size)
    load_tensor_float8(file, scale_name, weight_scale, scale_size)

    var weight_scale_2 = load_tensor_float32_scalar(file, scale2_name)
    var input_scale = load_tensor_float32_scalar(file, input_scale_name)

    return NvFp4Linear(
        weight=weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2,
        input_scale=input_scale,
        out_features=out_features,
        in_features=in_features,
    )


fn load_nvfp4_expert(
    file: SafetensorFile,
    prefix: String,  # e.g., "model.layers.0.mlp.experts.0"
) -> NvFp4Expert:
    """Load a complete MoE expert with NVFP4 projections."""
    var gate = load_nvfp4_linear(file, prefix + ".gate_proj")
    var up = load_nvfp4_linear(file, prefix + ".up_proj")
    var down = load_nvfp4_linear(file, prefix + ".down_proj")

    return NvFp4Expert(gate_proj=gate, up_proj=up, down_proj=down)


fn load_nvfp4_moe_layer(
    files: List[SafetensorFile],  # Multiple shards
    layer_idx: Int,
    num_experts: Int = 128,
    num_experts_per_tok: Int = 8,
    hidden_size: Int = 2048,
    intermediate_size: Int = 768,
) -> NvFp4MoELayer:
    """Load a complete MoE layer with all experts.

    For Qwen3-30B-A3B:
    - 128 experts
    - Each expert has gate_proj, up_proj, down_proj
    - Plus shared router_weight

    Args:
        files: List of safetensor files (model may be sharded)
        layer_idx: Layer index (0-47 for Qwen3-30B-A3B)
        num_experts: Total number of experts (128)
        num_experts_per_tok: Experts per token (8)
        hidden_size: Model hidden dimension (2048)
        intermediate_size: Expert intermediate dimension (768)
    """
    var prefix = "model.layers." + String(layer_idx) + ".mlp"

    # Allocate expert array
    var experts = UnsafePointer[NvFp4Expert].alloc(num_experts)

    # Load each expert
    # Experts may be spread across multiple safetensor shards
    for e in range(num_experts):
        var expert_prefix = prefix + ".experts." + String(e)

        # Find which file contains this expert
        for file in files:
            var first_tensor = expert_prefix + ".gate_proj.weight"
            if first_tensor in file.tensor_names:
                experts[e] = load_nvfp4_expert(file[], expert_prefix)
                break

    # Load router weight (not quantized)
    var router_size = num_experts * hidden_size
    var router_weight = UnsafePointer[Float32].alloc(router_size)

    var router_name = prefix + ".gate.weight"
    for file in files:
        if router_name in file.tensor_names:
            # Load bfloat16 router weight and convert to float32
            # TODO: Implement proper loading
            break

    return NvFp4MoELayer(
        experts=experts,
        router_weight=router_weight,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


struct NvFp4Model:
    """Complete NVFP4 quantized model."""
    var embed_tokens: UnsafePointer[BFloat16]  # Not quantized
    var layers: List[NvFp4MoELayer]
    var lm_head: UnsafePointer[BFloat16]  # Not quantized (excluded)
    var norm: UnsafePointer[BFloat16]  # Not quantized

    var config: NvFp4QuantConfig
    var vocab_size: Int
    var hidden_size: Int
    var num_layers: Int

    fn __init__(out self):
        self.embed_tokens = UnsafePointer[BFloat16]()
        self.layers = List[NvFp4MoELayer]()
        self.lm_head = UnsafePointer[BFloat16]()
        self.norm = UnsafePointer[BFloat16]()
        self.config = NvFp4QuantConfig()
        self.vocab_size = 151936
        self.hidden_size = 2048
        self.num_layers = 48


fn load_nvfp4_model(model_path: String) -> NvFp4Model:
    """Load a complete NVFP4 quantized model from Hugging Face format.

    Args:
        model_path: Path to model directory containing:
            - config.json
            - hf_quant_config.json
            - model-*.safetensors
    """
    var model = NvFp4Model()

    # Parse quantization config
    var quant_config_path = model_path + "/hf_quant_config.json"
    # TODO: Read and parse JSON file
    # model.config = parse_quant_config(read_file(quant_config_path))

    # Find all safetensor shards
    var files = List[SafetensorFile]()
    # TODO: Glob model-*.safetensors files
    # For Qwen3-30B-A3B: model-00001-of-00004.safetensors to model-00004-of-00004.safetensors

    # Load embeddings (bfloat16, not quantized)
    # model.embed_tokens = load_tensor_bfloat16(files[0], "model.embed_tokens.weight")

    # Load MoE layers
    for layer_idx in range(model.num_layers):
        var moe_layer = load_nvfp4_moe_layer(
            files,
            layer_idx,
            num_experts=128,
            num_experts_per_tok=8,
            hidden_size=2048,
            intermediate_size=768,
        )
        model.layers.append(moe_layer)

    # Load final norm and lm_head (not quantized)
    # model.norm = ...
    # model.lm_head = ...

    return model
