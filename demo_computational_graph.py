"""
Interactive demonstration of how computational graphs route gradients.

This script shows concrete examples of how a single loss value can produce
different gradients for different parameters based on the computational graph.
"""

import jax
import jax.numpy as jnp
import numpy as np


def example_1_independent_losses():
    """Example 1: Two independent losses combined into one."""
    print("=" * 80)
    print("EXAMPLE 1: Independent Losses")
    print("=" * 80)
    
    # Two separate parameters
    params_A = jnp.array(3.0)
    params_B = jnp.array(4.0)
  
    def compute_loss(p_A, p_B):
        loss_A = p_A ** 2      # Only depends on params_A
        loss_B = p_B ** 2      # Only depends on params_B
        total = loss_A + loss_B  # Single number!
        return total
    
    # Compute gradients
    grad_fn = jax.grad(compute_loss, argnums=(0, 1))
    grad_A, grad_B = grad_fn(params_A, params_B)
    
    total_loss = compute_loss(params_A, params_B)
    
    print(f"\nParameters:")
    print(f"  params_A = {float(params_A)}")
    print(f"  params_B = {float(params_B)}")
    
    print(f"\nLoss Components:")
    print(f"  loss_A = params_A² = {float(params_A ** 2)}")
    print(f"  loss_B = params_B² = {float(params_B ** 2)}")
    print(f"  total_loss = {float(total_loss)} (just a single number!)")
    
    print(f"\nGradients:")
    print(f"  ∂total_loss/∂params_A = {float(grad_A)} (= 2 × params_A)")
    print(f"  ∂total_loss/∂params_B = {float(grad_B)} (= 2 × params_B)")
    
    print(f"\n✓ Even though total_loss is a single number, JAX computed different")
    print(f"  gradients for different parameters based on the computational graph!")


def example_2_stop_gradient():
    """Example 2: Using stop_gradient to block gradient flow."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: stop_gradient Blocks Gradient Flow")
    print("=" * 80)
    
    params_A = jnp.array(3.0)
    params_B = jnp.array(4.0)
    
    def compute_loss_without_stop_grad(p_A, p_B):
        x = p_A * p_B          # Connects both parameters
        loss = x ** 2          # Depends on both p_A and p_B
        return loss
    
    def compute_loss_with_stop_grad(p_A, p_B):
        x = p_A * p_B
        x_detached = jax.lax.stop_gradient(x)  # Break the connection!
        loss = x_detached ** 2  # No longer depends on p_A or p_B
        return loss
    
    # Without stop_gradient
    grad_A_no_stop, grad_B_no_stop = jax.grad(compute_loss_without_stop_grad, argnums=(0, 1))(params_A, params_B)
    
    # With stop_gradient
    grad_A_with_stop, grad_B_with_stop = jax.grad(compute_loss_with_stop_grad, argnums=(0, 1))(params_A, params_B)
    
    print(f"\nWithout stop_gradient:")
    print(f"  loss = (params_A × params_B)²")
    print(f"  ∂loss/∂params_A = {float(grad_A_no_stop)}")
    print(f"  ∂loss/∂params_B = {float(grad_B_no_stop)}")
    print(f"  Both get gradients! ✓")
    
    print(f"\nWith stop_gradient:")
    print(f"  x = params_A × params_B")
    print(f"  x_detached = stop_gradient(x)")
    print(f"  loss = x_detached²")
    print(f"  ∂loss/∂params_A = {float(grad_A_with_stop)}")
    print(f"  ∂loss/∂params_B = {float(grad_B_with_stop)}")
    print(f"  Both get ZERO gradients! ✓")
    
    print(f"\n✓ stop_gradient creates a barrier in the computational graph!")


def example_3_ki_style_loss():
    """Example 3: KI-style dual loss with gradient isolation."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: KI-Style Gradient Isolation")
    print("=" * 80)
    
    # Simulate VLM and Action Expert parameters
    vlm_params = jnp.array([1.0, 2.0, 3.0])
    action_params = jnp.array([4.0, 5.0])
    
    def compute_ki_loss(vlm_p, action_p):
        # Path 1: VLM processes input
        vlm_output = jnp.sum(vlm_p ** 2)  # Depends on vlm_params
        
        # Compute FAST loss (depends on VLM)
        fast_loss = vlm_output * 2.0
        
        # Stop gradient: prevent action loss from affecting VLM
        vlm_output_detached = jax.lax.stop_gradient(vlm_output)
        
        # Path 2: Action expert processes (depends on action_params AND detached VLM output)
        action_output = jnp.sum(action_p ** 2) + vlm_output_detached
        
        # Compute action loss
        action_loss = action_output * 3.0
        
        # Combined loss (single number!)
        total_loss = action_loss + fast_loss
        
        return total_loss, action_loss, fast_loss
    
    # Compute gradients
    grad_fn = jax.grad(lambda v, a: compute_ki_loss(v, a)[0], argnums=(0, 1))
    vlm_grads, action_grads = grad_fn(vlm_params, action_params)
    
    total, action_loss, fast_loss = compute_ki_loss(vlm_params, action_params)
    
    print(f"\nLoss Components:")
    print(f"  action_loss = {float(action_loss):.2f}")
    print(f"  fast_loss = {float(fast_loss):.2f}")
    print(f"  total_loss = {float(total):.2f} (single number!)")
    
    print(f"\nVLM Parameter Gradients:")
    print(f"  {vlm_grads}")
    print(f"  Sum: {float(jnp.sum(vlm_grads)):.4f}")
    print(f"  Non-zero? {jnp.any(vlm_grads != 0)}")
    
    print(f"\nAction Expert Parameter Gradients:")
    print(f"  {action_grads}")
    print(f"  Sum: {float(jnp.sum(action_grads)):.4f}")
    print(f"  Non-zero? {jnp.any(action_grads != 0)}")
    
    print(f"\n✓ VLM gets gradients from fast_loss only (action path blocked by stop_gradient)")
    print(f"✓ Action expert gets gradients from action_loss only")
    print(f"✓ Both trained from single combined loss!")


def example_4_verify_gradient_isolation():
    """Example 4: Verify gradients are truly isolated."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Verifying Gradient Isolation")
    print("=" * 80)
    
    vlm_params = jnp.array(2.0)
    action_params = jnp.array(3.0)
    
    def compute_action_loss_only(vlm_p, action_p):
        vlm_output = vlm_p ** 2
        vlm_output_detached = jax.lax.stop_gradient(vlm_output)
        action_output = action_p ** 2 + vlm_output_detached
        return action_output
    
    def compute_fast_loss_only(vlm_p, action_p):
        vlm_output = vlm_p ** 2
        return vlm_output
    
    def compute_combined_loss(vlm_p, action_p):
        action_loss = compute_action_loss_only(vlm_p, action_p)
        fast_loss = compute_fast_loss_only(vlm_p, action_p)
        return action_loss + fast_loss
    
    # Compute gradients of each component separately
    vlm_grad_action, action_grad_action = jax.grad(compute_action_loss_only, argnums=(0, 1))(vlm_params, action_params)
    vlm_grad_fast, action_grad_fast = jax.grad(compute_fast_loss_only, argnums=(0, 1))(vlm_params, action_params)
    
    # Compute gradients of combined loss
    vlm_grad_combined, action_grad_combined = jax.grad(compute_combined_loss, argnums=(0, 1))(vlm_params, action_params)
    
    print(f"\nGradients from action_loss only:")
    print(f"  ∂action_loss/∂vlm_params = {float(vlm_grad_action)} (should be 0!)")
    print(f"  ∂action_loss/∂action_params = {float(action_grad_action)}")
    
    print(f"\nGradients from fast_loss only:")
    print(f"  ∂fast_loss/∂vlm_params = {float(vlm_grad_fast)}")
    print(f"  ∂fast_loss/∂action_params = {float(action_grad_fast)} (should be 0!)")
    
    print(f"\nGradients from combined loss:")
    print(f"  ∂combined/∂vlm_params = {float(vlm_grad_combined)}")
    print(f"  ∂combined/∂action_params = {float(action_grad_combined)}")
    
    print(f"\nVerification:")
    print(f"  vlm_grad_combined ≈ vlm_grad_fast? {jnp.allclose(vlm_grad_combined, vlm_grad_fast)} ✓")
    print(f"  action_grad_combined ≈ action_grad_action? {jnp.allclose(action_grad_combined, action_grad_action)} ✓")
    
    print(f"\n✓ Combined loss produces isolated gradients!")
    print(f"✓ VLM only receives fast_loss gradients")
    print(f"✓ Action expert only receives action_loss gradients")


def visualize_graph():
    """Visualize the computational graph structure."""
    print("\n" + "=" * 80)
    print("CONCEPTUAL GRAPH VISUALIZATION")
    print("=" * 80)
    
    print("""
    total_loss = action_loss + fast_loss
         │              
         ├──────────────────┬──────────────────┐
         │                  │                  │
    [combined]              │                  │
         │                  │                  │
         │            action_loss          fast_loss
         │                  │                  │
         │                  │                  │
    optimizer              │                  │
         │                  │                  │
         ▼                  ▼                  ▼
    [computes          action_output    vlm_output
     gradients]            │                  │
         │                 │                  │
         │                 │                  │
    Backprop:              │                  │
         │            action_params      vlm_params
         │                 ▲                  ▲
         │                 │                  │
         │                 │                  │
    Gradient flow:         │                  │
    ─────────────>    [✓ receives      [✓ receives
                       gradients]       gradients]
                           │                  │
                           │                  │
    From action_loss ─────>│                  │
                                               │
    From fast_loss ───────────────────────────>│
    
    NOTE: stop_gradient on vlm_output_detached blocks:
          action_loss → vlm_params (shown by ╳)
    """)


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "Computational Graph Magic Demonstrated" + " " * 25 + "║")
    print("╚" + "=" * 78 + "╝")
    
    example_1_independent_losses()
    example_2_stop_gradient()
    example_3_ki_style_loss()
    example_4_verify_gradient_isolation()
    visualize_graph()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
    The "magic" of gradient routing comes from:
    
    1. JAX builds a computational graph as you compute the loss
    2. Each operation records which parameters it depends on
    3. stop_gradient creates barriers that block gradient flow
    4. Backpropagation traverses the graph to compute gradients
    5. Parameters only get gradients from paths that connect to them
    
    The loss is a single number, but the GRAPH determines which parameters
    get which gradients!
    
    In KI:
    - total_loss = action_loss + FAST_loss (single number)
    - But VLM only receives gradients from FAST_loss path
    - And action expert only receives gradients from action_loss path
    - stop_gradient on KV cache ensures this isolation
    """)
    print("=" * 80)
