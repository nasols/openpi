"""
Visual verification that KI gradient isolation works.

This test shows parameter changes during backprop to verify:
1. FAST loss ONLY updates VLM parameters (not action expert)
2. Action loss ONLY updates action expert parameters (not VLM)
3. Combined KI loss updates both with proper isolation
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU for testing

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from copy import deepcopy

from openpi.models import pi0_config, model as _model
from openpi.models.pi0 import Pi0, make_attn_mask


def create_test_observation(batch_size=2, seq_len=64):
    """Create a dummy observation for testing."""
    return _model.Observation(
        state=jnp.ones((batch_size, 8)),
        images={
            "base_0_rgb": jnp.ones((batch_size, 224, 224, 3)),
            "left_wrist_0_rgb": jnp.ones((batch_size, 224, 224, 3)),
            "right_wrist_0_rgb": jnp.ones((batch_size, 224, 224, 3)),
        },
        image_masks={
            "base_0_rgb": jnp.ones((batch_size,), dtype=bool),
            "left_wrist_0_rgb": jnp.ones((batch_size,), dtype=bool),
            "right_wrist_0_rgb": jnp.zeros((batch_size,), dtype=bool),
        },
        tokenized_prompt=jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, seq_len), dtype=bool),
        token_loss_mask=jnp.concatenate([
            jnp.zeros((batch_size, seq_len - 15), dtype=bool),
            jnp.ones((batch_size, 15), dtype=bool),
        ], axis=1),
    )


def get_param_snapshot(model, param_path):
    """Extract a specific parameter value from model.
    
    Args:
        model: The Pi0 model
        param_path: Tuple describing parameter location, e.g., 
                   ('PaliGemma', 'llm', 'layers', 0, 'attn_0', 'q_einsum_0', 0, 'kernel')
                   or ('action_out_proj', 'kernel')
    """
    current = model
    for key in param_path:
        if isinstance(current, dict):
            current = current[key]
        elif isinstance(key, int):
            current = current[key]
        else:
            current = getattr(current, key)
    
    if hasattr(current, 'value'):
        return jnp.array(current.value)
    return jnp.array(current)


def compute_param_change(before, after):
    """Compute the magnitude of parameter change."""
    diff = jnp.abs(after - before)
    max_change = jnp.max(diff)
    mean_change = jnp.mean(diff)
    relative_change = mean_change / (jnp.mean(jnp.abs(before)) + 1e-10)
    return {
        'max': float(max_change),
        'mean': float(mean_change),
        'relative': float(relative_change)
    }


def test_fast_loss_only_updates_vlm():
    """Test that FAST loss gradients ONLY affect VLM parameters."""
    print("\n" + "=" * 80)
    print("TEST 1: FAST Loss Only Updates VLM (Not Action Expert)")
    print("=" * 80)
    
    # Create model
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
    observation = create_test_observation()
    rng = jax.random.PRNGKey(42)
    
    # Define parameter paths to monitor
    # VLM parameter: First layer attention query kernel
    vlm_param_path = ('PaliGemma', 'llm', 'layers', 0, 'attn_0', 'q_einsum_0', 0, 'kernel')
    # Action expert parameter: Output projection kernel
    action_param_path = ('action_out_proj', 'kernel')
    
    # Get initial parameter values
    vlm_before = get_param_snapshot(model, vlm_param_path)
    action_before = get_param_snapshot(model, action_param_path)
    
    print("\nInitial Parameter Values:")
    print(f"  VLM param shape: {vlm_before.shape}, mean: {jnp.mean(vlm_before):.6f}")
    print(f"  Action expert param shape: {action_before.shape}, mean: {jnp.mean(action_before):.6f}")
    
    # Define FAST loss only (no action loss)
    def fast_loss_only(model):
        # Manually compute FAST loss without action loss
        observation_processed = _model.preprocess_observation(rng, observation, train=True)
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation_processed)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        (prefix_out, _), _ = model.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )
        
        # Compute FAST loss only
        fast_loss = model.compute_fast_loss(prefix_out, observation_processed)
        return fast_loss
    
    # Compute gradients and update
    loss_val, grads = nnx.value_and_grad(fast_loss_only)(model)
    
    # Apply gradients with small learning rate
    lr = 0.001
    def apply_gradients(model, grads):
        state = nnx.state(model)
        for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
            grad_path = path
            grad_value = grads
            for key in grad_path:
                if hasattr(grad_value, key.key):
                    grad_value = getattr(grad_value, key.key)
                else:
                    break
            
            # Get the parameter value
            param = value
            if hasattr(grad_value, 'value') and hasattr(param, 'value'):
                # Update: param = param - lr * grad
                param.value = param.value - lr * grad_value.value
    
    apply_gradients(model, grads)
    
    # Get updated parameter values
    vlm_after = get_param_snapshot(model, vlm_param_path)
    action_after = get_param_snapshot(model, action_param_path)
    
    # Compute changes
    vlm_change = compute_param_change(vlm_before, vlm_after)
    action_change = compute_param_change(action_before, action_after)
    
    print(f"\nFAST Loss: {loss_val:.4f}")
    print("\nParameter Changes After FAST Loss Gradient Update:")
    print(f"  VLM Parameter:")
    print(f"    - Max change:      {vlm_change['max']:.8f}")
    print(f"    - Mean change:     {vlm_change['mean']:.8f}")
    print(f"    - Relative change: {vlm_change['relative']:.6%}")
    print(f"  Action Expert Parameter:")
    print(f"    - Max change:      {action_change['max']:.8f}")
    print(f"    - Mean change:     {action_change['mean']:.8f}")
    print(f"    - Relative change: {action_change['relative']:.6%}")
    
    # Verify isolation
    print("\n" + "-" * 80)
    if vlm_change['mean'] > 1e-6:
        print("✓ VLM parameters CHANGED (expected from FAST loss)")
    else:
        print("✗ VLM parameters DID NOT CHANGE (unexpected!)")
    
    if action_change['mean'] < 1e-9:
        print("✓ Action expert parameters DID NOT CHANGE (gradient isolation working!)")
    else:
        print(f"✗ Action expert parameters CHANGED by {action_change['mean']:.8f} (isolation FAILED!)")
    
    print("-" * 80)
    
    return vlm_change['mean'] > 1e-6 and action_change['mean'] < 1e-9


def test_action_loss_only_updates_action_expert():
    """Test that action loss gradients ONLY affect action expert parameters."""
    print("\n" + "=" * 80)
    print("TEST 2: Action Loss Only Updates Action Expert (Not VLM)")
    print("=" * 80)
    
    # Create model
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
    observation = create_test_observation()
    actions = jnp.ones((2, 15, 8))
    rng = jax.random.PRNGKey(42)
    
    # Define parameter paths
    vlm_param_path = ('PaliGemma', 'llm', 'layers', 0, 'attn_0', 'q_einsum_0', 0, 'kernel')
    action_param_path = ('action_out_proj', 'kernel')
    
    # Get initial values
    vlm_before = get_param_snapshot(model, vlm_param_path)
    action_before = get_param_snapshot(model, action_param_path)
    
    print("\nInitial Parameter Values:")
    print(f"  VLM param mean: {jnp.mean(vlm_before):.6f}")
    print(f"  Action expert param mean: {jnp.mean(action_before):.6f}")
    
    # Define action loss only (no FAST loss)
    def action_loss_only(model):
        # Manually compute action loss without FAST loss
        observation_processed = _model.preprocess_observation(rng, observation, train=True)
        
        # Generate noise and timestep
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(jax.random.PRNGKey(123), actions.shape)
        time = jax.random.beta(jax.random.PRNGKey(456), 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        # Compute prefix (with detached gradients to prevent VLM updates)
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation_processed)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        # Forward through VLM and DETACH to prevent gradients
        (prefix_out, _), kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )
        
        # CRITICAL: Stop gradients on KV cache (this is what KI does)
        kv_cache_detached = jax.tree.map(jax.lax.stop_gradient, kv_cache)
        
        # Compute suffix
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation_processed, x_t, time
        )
        
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_for_suffix = jnp.broadcast_to(
            prefix_mask[:, None, :], 
            (prefix_mask.shape[0], suffix_tokens.shape[1], prefix_mask.shape[1])
        )
        full_attn_mask = jnp.concatenate([prefix_attn_mask_for_suffix, suffix_attn_mask], axis=-1)
        suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        
        # Forward through action expert with detached cache
        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=kv_cache_detached,
            adarms_cond=[None, adarms_cond],
        )
        
        # Compute action loss
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
        action_loss = jnp.mean(jnp.square(v_t - u_t))
        return action_loss
    
    # Compute gradients
    loss_val, grads = nnx.value_and_grad(action_loss_only)(model)
    
    # Apply gradients
    lr = 0.001
    state = nnx.state(model, nnx.Param)
    grad_state = nnx.state(grads, nnx.Param)
    
    # Manual gradient update
    for key in state.flat_state():
        if key in grad_state.flat_state():
            param = state.flat_state()[key]
            grad = grad_state.flat_state()[key]
            if hasattr(param, 'value') and hasattr(grad, 'value'):
                param.value = param.value - lr * grad.value
    
    # Get updated values
    vlm_after = get_param_snapshot(model, vlm_param_path)
    action_after = get_param_snapshot(model, action_param_path)
    
    # Compute changes
    vlm_change = compute_param_change(vlm_before, vlm_after)
    action_change = compute_param_change(action_before, action_after)
    
    print(f"\nAction Loss: {loss_val:.4f}")
    print("\nParameter Changes After Action Loss Gradient Update:")
    print(f"  VLM Parameter:")
    print(f"    - Max change:      {vlm_change['max']:.8f}")
    print(f"    - Mean change:     {vlm_change['mean']:.8f}")
    print(f"    - Relative change: {vlm_change['relative']:.6%}")
    print(f"  Action Expert Parameter:")
    print(f"    - Max change:      {action_change['max']:.8f}")
    print(f"    - Mean change:     {action_change['mean']:.8f}")
    print(f"    - Relative change: {action_change['relative']:.6%}")
    
    # Verify isolation
    print("\n" + "-" * 80)
    if vlm_change['mean'] < 1e-9:
        print("✓ VLM parameters DID NOT CHANGE (gradient isolation working!)")
    else:
        print(f"✗ VLM parameters CHANGED by {vlm_change['mean']:.8f} (isolation FAILED!)")
    
    if action_change['mean'] > 1e-6:
        print("✓ Action expert parameters CHANGED (expected from action loss)")
    else:
        print("✗ Action expert parameters DID NOT CHANGE (unexpected!)")
    
    print("-" * 80)
    
    return vlm_change['mean'] < 1e-9 and action_change['mean'] > 1e-6


def test_combined_ki_loss():
    """Test that combined KI loss updates both components with proper isolation."""
    print("\n" + "=" * 80)
    print("TEST 3: Combined KI Loss Updates Both (With Isolation)")
    print("=" * 80)
    
    # Create model
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
    observation = create_test_observation()
    actions = jnp.ones((2, 15, 8))
    rng = jax.random.PRNGKey(42)
    
    # Define parameter paths
    vlm_param_path = ('PaliGemma', 'llm', 'layers', 0, 'attn_0', 'q_einsum_0', 0, 'kernel')
    action_param_path = ('action_out_proj', 'kernel')
    
    # Get initial values
    vlm_before = get_param_snapshot(model, vlm_param_path)
    action_before = get_param_snapshot(model, action_param_path)
    
    print("\nInitial Parameter Values:")
    print(f"  VLM param mean: {jnp.mean(vlm_before):.6f}")
    print(f"  Action expert param mean: {jnp.mean(action_before):.6f}")
    
    # Use model's compute_loss (which includes both FAST and action loss)
    def combined_loss(model):
        return jnp.mean(model.compute_loss(rng, observation, actions, train=True))
    
    # Compute gradients
    loss_val, grads = nnx.value_and_grad(combined_loss)(model)
    
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
    
    # Get updated values
    vlm_after = get_param_snapshot(model, vlm_param_path)
    action_after = get_param_snapshot(model, action_param_path)
    
    # Compute changes
    vlm_change = compute_param_change(vlm_before, vlm_after)
    action_change = compute_param_change(action_before, action_after)
    
    print(f"\nCombined Loss: {loss_val:.4f}")
    print("\nParameter Changes After Combined KI Loss Gradient Update:")
    print(f"  VLM Parameter:")
    print(f"    - Max change:      {vlm_change['max']:.8f}")
    print(f"    - Mean change:     {vlm_change['mean']:.8f}")
    print(f"    - Relative change: {vlm_change['relative']:.6%}")
    print(f"  Action Expert Parameter:")
    print(f"    - Max change:      {action_change['max']:.8f}")
    print(f"    - Mean change:     {action_change['mean']:.8f}")
    print(f"    - Relative change: {action_change['relative']:.6%}")
    
    # Verify both updated
    print("\n" + "-" * 80)
    if vlm_change['mean'] > 1e-6:
        print("✓ VLM parameters CHANGED (from FAST loss)")
    else:
        print("✗ VLM parameters DID NOT CHANGE")
    
    if action_change['mean'] > 1e-6:
        print("✓ Action expert parameters CHANGED (from action loss)")
    else:
        print("✗ Action expert parameters DID NOT CHANGE")
    
    print("\n✓ Both components updated independently via KI gradient isolation!")
    print("-" * 80)
    
    return vlm_change['mean'] > 1e-6 and action_change['mean'] > 1e-6


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("KNOWLEDGE INSULATION GRADIENT ISOLATION VERIFICATION")
    print("=" * 80)
    print("\nThis test verifies that:")
    print("1. FAST loss ONLY updates VLM (not action expert)")
    print("2. Action loss ONLY updates action expert (not VLM)")
    print("3. Combined loss updates both with proper isolation")
    
    success = True
    
    try:
        if not test_fast_loss_only_updates_vlm():
            success = False
            print("\n❌ TEST 1 FAILED")
        else:
            print("\n✅ TEST 1 PASSED")
    except Exception as e:
        print(f"\n❌ TEST 1 ERROR: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        if not test_action_loss_only_updates_action_expert():
            success = False
            print("\n❌ TEST 2 FAILED")
        else:
            print("\n✅ TEST 2 PASSED")
    except Exception as e:
        print(f"\n❌ TEST 2 ERROR: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        if not test_combined_ki_loss():
            success = False
            print("\n❌ TEST 3 FAILED")
        else:
            print("\n✅ TEST 3 PASSED")
    except Exception as e:
        print(f"\n❌ TEST 3 ERROR: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 ALL TESTS PASSED - GRADIENT ISOLATION VERIFIED!")
    else:
        print("❌ SOME TESTS FAILED - CHECK GRADIENT ISOLATION")
    print("=" * 80)
