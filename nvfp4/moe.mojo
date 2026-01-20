# ===----------------------------------------------------------------------=== #
# NVFP4 Mixture of Experts (MoE) Support for MAX
#
# Qwen3-30B-A3B uses:
#   - 128 total experts
#   - 8 experts active per token
#   - Each expert has: gate_proj, up_proj, down_proj (all NVFP4)
#
# This module provides:
#   1. Expert-wise weight storage with NVFP4 quantization
#   2. Token-to-expert routing
#   3. Fused MoE GEMM with NVFP4 dequantization
# ===----------------------------------------------------------------------=== #

from math import ceildiv
from memory import UnsafePointer

from .config import (
    NVFP4_GROUP_SIZE,
    NVFP4_PACK_FACTOR,
    NvFp4Config,
    NvFp4Linear,
)
from .nvfp4_gemm_gpu import nvfp4_gemm_kernel, BM, BN, BK


alias MAX_EXPERTS: Int = 128
alias TOP_K_EXPERTS: Int = 8  # Number of experts per token in Qwen3 MoE


struct NvFp4Expert:
    """A single MoE expert with NVFP4 quantized projections."""
    var gate_proj: NvFp4Linear  # [intermediate_size, hidden_size]
    var up_proj: NvFp4Linear    # [intermediate_size, hidden_size]
    var down_proj: NvFp4Linear  # [hidden_size, intermediate_size]

    fn __init__(
        out self,
        gate_proj: NvFp4Linear,
        up_proj: NvFp4Linear,
        down_proj: NvFp4Linear,
    ):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj


struct NvFp4MoELayer:
    """NVFP4 Mixture of Experts layer.

    For Qwen3-30B-A3B:
    - num_experts = 128
    - num_experts_per_tok = 8
    - hidden_size = 2048
    - moe_intermediate_size = 768 (per expert)
    """
    var experts: UnsafePointer[NvFp4Expert]
    var num_experts: Int
    var num_experts_per_tok: Int
    var hidden_size: Int
    var intermediate_size: Int

    # Router weights (not quantized)
    var router_weight: UnsafePointer[Float32]  # [num_experts, hidden_size]

    fn __init__(
        out self,
        experts: UnsafePointer[NvFp4Expert],
        router_weight: UnsafePointer[Float32],
        num_experts: Int,
        num_experts_per_tok: Int,
        hidden_size: Int,
        intermediate_size: Int,
    ):
        self.experts = experts
        self.router_weight = router_weight
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size


struct ExpertAssignment:
    """Token-to-expert assignment for a batch."""
    var expert_indices: UnsafePointer[Int32]    # [batch_size, top_k]
    var expert_weights: UnsafePointer[Float32]  # [batch_size, top_k]
    var batch_size: Int
    var top_k: Int


fn compute_router_logits(
    hidden_states: UnsafePointer[Float32],  # [batch_size, hidden_size]
    router_weight: UnsafePointer[Float32],   # [num_experts, hidden_size]
    logits: UnsafePointer[Float32],          # [batch_size, num_experts]
    batch_size: Int,
    hidden_size: Int,
    num_experts: Int,
):
    """Compute router logits: logits = hidden_states @ router_weight.T"""
    # Simple GEMM: [B, H] @ [E, H].T = [B, E]
    for b in range(batch_size):
        for e in range(num_experts):
            var sum: Float32 = 0.0
            for h in range(hidden_size):
                sum += hidden_states[b * hidden_size + h] * router_weight[e * hidden_size + h]
            logits[b * num_experts + e] = sum


fn top_k_softmax(
    logits: UnsafePointer[Float32],          # [batch_size, num_experts]
    assignment: ExpertAssignment,             # Output
    batch_size: Int,
    num_experts: Int,
    top_k: Int,
):
    """Select top-k experts per token with softmax weights.

    Uses normalized top-k selection: weights sum to 1 per token.
    """
    for b in range(batch_size):
        # Find top-k experts
        # Simple O(E*K) selection (can be optimized with heap)
        for k in range(top_k):
            var best_idx = -1
            var best_val: Float32 = -1e30

            for e in range(num_experts):
                # Skip already selected experts
                var already_selected = False
                for prev in range(k):
                    if assignment.expert_indices[b * top_k + prev] == e:
                        already_selected = True
                        break

                if not already_selected and logits[b * num_experts + e] > best_val:
                    best_val = logits[b * num_experts + e]
                    best_idx = e

            assignment.expert_indices[b * top_k + k] = best_idx

        # Compute softmax over selected experts
        var max_logit: Float32 = -1e30
        for k in range(top_k):
            var idx = assignment.expert_indices[b * top_k + k]
            var logit = logits[b * num_experts + idx]
            if logit > max_logit:
                max_logit = logit

        var sum_exp: Float32 = 0.0
        for k in range(top_k):
            var idx = assignment.expert_indices[b * top_k + k]
            var logit = logits[b * num_experts + idx]
            var exp_val = (logit - max_logit).exp()
            assignment.expert_weights[b * top_k + k] = exp_val
            sum_exp += exp_val

        # Normalize
        for k in range(top_k):
            assignment.expert_weights[b * top_k + k] /= sum_exp


fn nvfp4_moe_forward[
    hidden_size: Int,
    intermediate_size: Int,
    num_experts: Int,
    top_k: Int,
](
    moe: NvFp4MoELayer,
    hidden_states: UnsafePointer[Float32],  # [batch_size, hidden_size]
    output: UnsafePointer[Float32],          # [batch_size, hidden_size]
    batch_size: Int,
    # Workspace buffers
    router_logits: UnsafePointer[Float32],   # [batch_size, num_experts]
    intermediate_buffer: UnsafePointer[Float32],  # [batch_size * top_k, intermediate_size]
):
    """NVFP4 MoE forward pass.

    Flow:
    1. Compute router logits
    2. Select top-k experts per token
    3. For each selected expert:
       a. gate = gate_proj(hidden) * sigmoid
       b. up = up_proj(hidden)
       c. intermediate = gate * up (SiLU gating)
       d. expert_out = down_proj(intermediate)
    4. Combine expert outputs with routing weights
    """
    # Allocate assignment buffers
    var assignment = ExpertAssignment(
        expert_indices = UnsafePointer[Int32].alloc(batch_size * top_k),
        expert_weights = UnsafePointer[Float32].alloc(batch_size * top_k),
        batch_size = batch_size,
        top_k = top_k,
    )

    # Step 1: Compute router logits
    compute_router_logits(
        hidden_states,
        moe.router_weight,
        router_logits,
        batch_size,
        hidden_size,
        num_experts,
    )

    # Step 2: Top-k selection with softmax
    top_k_softmax(router_logits, assignment, batch_size, num_experts, top_k)

    # Initialize output to zero
    for i in range(batch_size * hidden_size):
        output[i] = 0.0

    # Step 3: Process each token's selected experts
    for b in range(batch_size):
        for k in range(top_k):
            var expert_idx = Int(assignment.expert_indices[b * top_k + k])
            var expert_weight = assignment.expert_weights[b * top_k + k]

            var expert = moe.experts[expert_idx]

            # Get token's hidden state
            var token_hidden = hidden_states + b * hidden_size

            # Intermediate buffers for this expert computation
            var gate_out = intermediate_buffer + (b * top_k + k) * intermediate_size
            var up_out = intermediate_buffer + (batch_size * top_k + b * top_k + k) * intermediate_size

            # gate_proj: [intermediate_size, hidden_size] @ [hidden_size, 1]
            nvfp4_gemm_kernel[intermediate_size, 1, hidden_size](
                expert.gate_proj.weight,
                expert.gate_proj.weight_scale,
                token_hidden,
                gate_out,
                expert.gate_proj.alpha(),
            )

            # up_proj: [intermediate_size, hidden_size] @ [hidden_size, 1]
            nvfp4_gemm_kernel[intermediate_size, 1, hidden_size](
                expert.up_proj.weight,
                expert.up_proj.weight_scale,
                token_hidden,
                up_out,
                expert.up_proj.alpha(),
            )

            # SiLU gating: gate_out = silu(gate_out) * up_out
            for i in range(intermediate_size):
                var g = gate_out[i]
                var sigmoid_g = 1.0 / (1.0 + (-g).exp())
                var silu_g = g * sigmoid_g
                gate_out[i] = silu_g * up_out[i]

            # down_proj: [hidden_size, intermediate_size] @ [intermediate_size, 1]
            var expert_output = UnsafePointer[Float32].alloc(hidden_size)
            nvfp4_gemm_kernel[hidden_size, 1, intermediate_size](
                expert.down_proj.weight,
                expert.down_proj.weight_scale,
                gate_out,
                expert_output,
                expert.down_proj.alpha(),
            )

            # Accumulate weighted expert output
            for i in range(hidden_size):
                output[b * hidden_size + i] += expert_weight * expert_output[i]

            expert_output.free()

    # Cleanup
    assignment.expert_indices.free()
    assignment.expert_weights.free()


# Fused MoE kernel for better performance
# This version groups tokens by their selected experts
fn nvfp4_fused_moe_forward[
    hidden_size: Int,
    intermediate_size: Int,
    num_experts: Int,
    top_k: Int,
](
    moe: NvFp4MoELayer,
    hidden_states: UnsafePointer[Float32],   # [batch_size, hidden_size]
    output: UnsafePointer[Float32],           # [batch_size, hidden_size]
    batch_size: Int,
    router_logits: UnsafePointer[Float32],   # [batch_size, num_experts]
):
    """Fused NVFP4 MoE forward with token grouping.

    Groups tokens by their selected experts to enable batched GEMM,
    which is more efficient than processing tokens individually.

    This is the optimized path for production use.
    """
    # TODO: Implement expert-grouped batching for better GPU utilization
    # For now, use the sequential version
    var intermediate_buffer = UnsafePointer[Float32].alloc(
        batch_size * top_k * intermediate_size * 2  # gate + up projections
    )

    nvfp4_moe_forward[hidden_size, intermediate_size, num_experts, top_k](
        moe,
        hidden_states,
        output,
        batch_size,
        router_logits,
        intermediate_buffer,
    )

    intermediate_buffer.free()
