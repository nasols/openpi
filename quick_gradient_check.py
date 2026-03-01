"""
Quick visual test to see parameter changes during KI training.

Run this to instantly verify gradient isolation is working.
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
from flax import nnx

from openpi.models import pi0_config, model as _model
from openpi.models.pi0 import Pi0


def quick_test():
    """Fast test showing parameter changes before/after gradient update."""
    
    print("\n" + "="*80)
    print("QUICK PARAMETER CHANGE CHECK")
    print("="*80)
    
    # Create tiny model
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=8,
        action_horizon=15,
        knowledge_insulation=True,
        ki_fast_loss_weight=1.0,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = Pi0(config, nnx.Rngs(0))
    
    # Create dummy data
    observation = _model.Observation(
        state=jnp.ones((2, 8)),
        images={
            "base_0_rgb": jnp.ones((2, 224, 224, 3)),
            "left_wrist_0_rgb": jnp.ones((2, 224, 224, 3)),
            "right_wrist_0_rgb": jnp.ones((2, 224, 224, 3)),
        },
        image_masks={
            "base_0_rgb": jnp.ones((2,), dtype=bool),
            "left_wrist_0_rgb": jnp.ones((2,), dtype=bool),
            "right_wrist_0_rgb": jnp.zeros((2,), dtype=bool),
        },
        tokenized_prompt=jnp.ones((2, 64), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((2, 64), dtype=bool),
        token_loss_mask=jnp.concatenate([
            jnp.zeros((2, 49), dtype=bool),
            jnp.ones((2, 15), dtype=bool),
        ], axis=1),
    )
    actions = jnp.ones((2, 15, 8))
    rng = jax.random.PRNGKey(42)
    
    # Snapshot BEFORE gradient update
    print("\n📸 Taking parameter snapshots BEFORE gradient update...")
    vlm_param_before = model.PaliGemma.llm.layers[0].attn_0.q_einsum_0[0].kernel.value
    action_param_before = model.action_out_proj.kernel.value
    
    vlm_mean_before = float(jnp.mean(vlm_param_before))
    action_mean_before = float(jnp.mean(action_param_before))
    
    print(f"VLM param mean:          {vlm_mean_before:.10f}")
    print(f"Action expert param mean: {action_mean_before:.10f}")
    
    # Compute loss and gradients
    print("\n🔄 Computing gradients...")
    def loss_fn(model):
        return jnp.mean(model.compute_loss(rng, observation, actions, train=True))
    
    loss_val, grads = nnx.value_and_grad(loss_fn)(model)
    print(f"Loss: {loss_val:.4f}")
    
    # Check if gradients exist
    vlm_grad = grads.PaliGemma.llm.layers[0].attn_0.q_einsum_0[0].kernel.value
    action_grad = grads.action_out_proj.kernel.value
    
    vlm_grad_norm = float(jnp.linalg.norm(vlm_grad))
    action_grad_norm = float(jnp.linalg.norm(action_grad))
    
    print(f"\nVLM gradient norm:          {vlm_grad_norm:.10f}")
    print(f"Action expert gradient norm: {action_grad_norm:.10f}")
    
    # Apply gradients (small learning rate)
    print("\n⬇️  Applying gradients (lr=0.001)...")
    lr = 0.001
    
    state = nnx.state(model, nnx.Param)
    grad_state = nnx.state(grads, nnx.Param)
    
    for key in state.flat_state():
        if key in grad_state.flat_state():
            param = state.flat_state()[key]
            grad = grad_state.flat_state()[key]
            if hasattr(param, 'value') and hasattr(grad, 'value'):
                param.value = param.value - lr * grad.value
    
    # Snapshot AFTER gradient update
    print("\n📸 Taking parameter snapshots AFTER gradient update...")
    vlm_param_after = model.PaliGemma.llm.layers[0].attn_0.q_einsum_0[0].kernel.value
    action_param_after = model.action_out_proj.kernel.value
    
    vlm_mean_after = float(jnp.mean(vlm_param_after))
    action_mean_after = float(jnp.mean(action_param_after))
    
    print(f"VLM param mean:          {vlm_mean_after:.10f}")
    print(f"Action expert param mean: {action_mean_after:.10f}")
    
    # Calculate changes
    vlm_change = abs(vlm_mean_after - vlm_mean_before)
    action_change = abs(action_mean_after - action_mean_before)
    
    vlm_max_change = float(jnp.max(jnp.abs(vlm_param_after - vlm_param_before)))
    action_max_change = float(jnp.max(jnp.abs(action_param_after - action_param_before)))
    
    print("\n" + "="*80)
    print("📊 PARAMETER CHANGES")
    print("="*80)
    print(f"\nVLM Parameter:")
    print(f"  Mean change:  {vlm_change:.10f}")
    print(f"  Max change:   {vlm_max_change:.10f}")
    print(f"  Did it change? {'YES ✓' if vlm_change > 1e-8 else 'NO ✗'}")
    
    print(f"\nAction Expert Parameter:")
    print(f"  Mean change:  {action_change:.10f}")
    print(f"  Max change:   {action_max_change:.10f}")
    print(f"  Did it change? {'YES ✓' if action_change > 1e-8 else 'NO ✗'}")
    
    # Verification
    print("\n" + "="*80)
    print("🔍 ISOLATION VERIFICATION")
    print("="*80)
    
    both_changed = vlm_change > 1e-8 and action_change > 1e-8
    
    if both_changed:
        print("✅ BOTH parameters changed (expected for combined KI loss)")
        print("\n   This is CORRECT because:")
        print("   - VLM gets gradients from FAST loss")
        print("   - Action expert gets gradients from action loss")
        print("   - KV cache detachment prevents cross-contamination")
        return True
    else:
        print("❌ NOT both parameters changed (unexpected!)")
        if vlm_change < 1e-8:
            print("   - VLM did NOT change (should change from FAST loss)")
        if action_change < 1e-8:
            print("   - Action expert did NOT change (should change from action loss)")
        return False


def test_action_loss_isolation():
    """Test that action loss doesn't affect VLM."""
    
    print("\n" + "="*80)
    print("TESTING: Action Loss Should NOT Update VLM")
    print("="*80)
    
    from openpi.models.pi0 import make_attn_mask
    
    # Create model
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=8,
        action_horizon=15,
        knowledge_insulation=True,
        ki_fast_loss_weight=0.0,  # NO FAST LOSS
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = Pi0(config, nnx.Rngs(0))
    
    # Create data
    observation = _model.Observation(
        state=jnp.ones((2, 8)),
        images={
            "base_0_rgb": jnp.ones((2, 224, 224, 3)),
            "left_wrist_0_rgb": jnp.ones((2, 224, 224, 3)),
            "right_wrist_0_rgb": jnp.ones((2, 224, 224, 3)),
        },
        image_masks={
            "base_0_rgb": jnp.ones((2,), dtype=bool),
            "left_wrist_0_rgb": jnp.ones((2,), dtype=bool),
            "right_wrist_0_rgb": jnp.zeros((2,), dtype=bool),
        },
        tokenized_prompt=jnp.ones((2, 64), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((2, 64), dtype=bool),
        token_loss_mask=jnp.zeros((2, 64), dtype=bool),  # NO FAST TOKENS
    )
    actions = jnp.ones((2, 15, 8))
    rng = jax.random.PRNGKey(42)
    
    # Get VLM param before
    vlm_before = model.PaliGemma.llm.layers[0].attn_0.q_einsum_0[0].kernel.value.copy()
    action_before = model.action_out_proj.kernel.value.copy()
    
    # Compute only action loss
    def loss_fn(model):
        return jnp.mean(model.compute_loss(rng, observation, actions, train=True))
    
    loss_val, grads = nnx.value_and_grad(loss_fn)(model)
    
    print(f"\nAction-only loss: {loss_val:.4f}")
    
    # Apply gradients
    lr = 0.001
    state = nnx.state(model, nnx.Param)
    grad_state = nnx.state(grads, nnx.Param)
    
    for key in state.flat_state():
        if key in grad_state.flat_state():
            param = state.flat_state()[key]
            grad = grad_state.flat_state()[key]
            if hasattr(param, 'value') and hasattr(grad, 'value'):
                param.value = param.value - lr * grad.value
    
    # Check changes
    vlm_after = model.PaliGemma.llm.layers[0].attn_0.q_einsum_0[0].kernel.value
    action_after = model.action_out_proj.kernel.value
    
    vlm_change = float(jnp.max(jnp.abs(vlm_after - vlm_before)))
    action_change = float(jnp.max(jnp.abs(action_after - action_before)))
    
    print(f"\nVLM max change:          {vlm_change:.10f}")
    print(f"Action expert max change: {action_change:.10f}")
    
    print("\n" + "="*80)
    if vlm_change < 1e-9:
        print("✅ ISOLATION WORKS: VLM did NOT change from action loss!")
        result = True
    else:
        print(f"❌ ISOLATION FAILED: VLM changed by {vlm_change:.10f}!")
        result = False
    
    if action_change > 1e-6:
        print("✅ Action expert DID change (expected)")
    else:
        print("❌ Action expert did NOT change (unexpected)")
        result = False
    
    print("="*80)
    return result


if __name__ == "__main__":
    print("\n" + "🔬"*40)
    print("KI GRADIENT ISOLATION - PARAMETER CHANGE VERIFICATION")
    print("🔬"*40)
    
    # Test 1: Both should update with combined loss
    success1 = quick_test()
    
    # Test 2: VLM should NOT update with action-only loss
    success2 = test_action_loss_isolation()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("\nGradient isolation is WORKING:")
        print("  ✓ Combined loss updates both VLM and action expert")
        print("  ✓ Action loss does NOT leak into VLM (via detached KV cache)")
        print("  ✓ FAST loss updates VLM (tested implicitly)")
    else:
        print("❌ SOME TESTS FAILED")
        if not success1:
            print("  ✗ Combined loss test failed")
        if not success2:
            print("  ✗ Action loss isolation test failed")
    print("="*80 + "\n")
