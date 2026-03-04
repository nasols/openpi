# The "Magic" of Computational Graphs: How One Loss Trains Different Parameters

## The Core Question

> "The loss is just a single number (14.6). How does JAX know which parts of the model to update with which parts of the loss?"

**Short Answer:** The loss is a number, but it's computed through a **computational graph** that tracks *how* it was created. JAX uses this graph to compute gradients, and `stop_gradient` creates barriers that isolate gradient flow.

---

## The Illusion: "Just a Number"

When you see:
```python
total_loss = 14.6
```

It looks like just a scalar. But in JAX, it's actually:

```python
total_loss = TracedValue(
    value=14.6,
    recipe=[
        "Add output from operation_1 (action_loss) and operation_2 (FAST_loss)",
        "operation_1 was computed from: suffix_out → action_proj → mse with u_t",
        "operation_2 was computed from: prefix_out → embedding_table → cross_entropy",
        "suffix_out depends on: action_expert_params (via KV cache with stop_grad)",
        "prefix_out depends on: prefix_tokens → VLM_params",
        ...
    ]
)
```

JAX **tracks every operation** that created that number, forming a graph.

---

## Computational Graph Example

### Simplified KI Forward Pass

```python
# What you write:
prefix_out, kv_cache = VLM(prefix_tokens)
FAST_loss = cross_entropy(prefix_out, targets)

kv_cache_detached = stop_gradient(kv_cache)
suffix_out = ActionExpert(suffix_tokens, kv_cache_detached)
action_loss = mse(suffix_out, actions)

total_loss = action_loss + FAST_loss  # Just a number (14.6)
```

### What JAX Actually Tracks

```
total_loss (14.6)
    │
    ├─── action_loss (1.5)
    │      │
    │      └─── mse(suffix_out, actions)
    │             │
    │             └─── suffix_out
    │                    │
    │                    └─── ActionExpert(suffix_tokens, kv_cache_detached)
    │                           │                          │
    │                           │                          └─── [STOP_GRADIENT BARRIER]
    │                           │                                    │
    │                           │                                    └─── kv_cache
    │                           │                                           │
    │                           │                                           └─── VLM(prefix_tokens)
    │                           │                                                  │
    │                           │                                                  └─── VLM_params ❌ (blocked!)
    │                           │
    │                           └─── ActionExpert_params ✓
    │
    └─── FAST_loss (13.1)
           │
           └─── cross_entropy(prefix_out, targets)
                  │
                  └─── prefix_out
                         │
                         └─── VLM(prefix_tokens)
                                │
                                └─── VLM_params ✓
```

---

## How Backpropagation Works

### Step 1: Forward Pass (Build the Graph)

```python
# Each operation creates a node in the graph
x1 = VLM(prefix_tokens)           # Node: depends on VLM_params
x2 = cross_entropy(x1, targets)   # Node: depends on x1
x3 = stop_gradient(kv_cache)      # Node: BREAKS gradient flow
x4 = ActionExpert(tokens, x3)     # Node: depends on ActionExpert_params, NOT on kv_cache grads
x5 = mse(x4, actions)             # Node: depends on x4
total = x5 + x2                   # Node: depends on x5 and x2
```

### Step 2: Backward Pass (Compute Gradients)

JAX traverses the graph **backwards**, computing derivatives using the chain rule:

```python
# Start with: ∂total_loss/∂total_loss = 1.0
∂L/∂total = 1.0

# Backprop through addition: total = x5 + x2
∂L/∂x5 = ∂L/∂total × ∂total/∂x5 = 1.0 × 1.0 = 1.0  # (action_loss branch)
∂L/∂x2 = ∂L/∂total × ∂total/∂x2 = 1.0 × 1.0 = 1.0  # (FAST_loss branch)

# Follow x2 branch (FAST_loss):
∂L/∂x1 = ∂L/∂x2 × ∂x2/∂x1 = 1.0 × (cross_entropy grad) = ...
∂L/∂VLM_params = ∂L/∂x1 × ∂x1/∂VLM_params = ... × (VLM forward grad)
# Result: VLM_params get gradients! ✓

# Follow x5 branch (action_loss):
∂L/∂x4 = ∂L/∂x5 × ∂x5/∂x4 = 1.0 × (mse grad) = ...
∂L/∂ActionExpert_params = ∂L/∂x4 × ∂x4/∂ActionExpert_params = ...
# Result: ActionExpert_params get gradients! ✓

# Try to backprop through x3 (stop_gradient):
∂L/∂x3 = ∂L/∂x4 × ∂x4/∂x3 = ... × [STOP_GRADIENT RETURNS ZERO]
# The gradient is STOPPED here!

# Try to reach VLM_params through x3:
∂L/∂VLM_params_via_x3 = ∂L/∂x3 × ... = 0 × ... = 0
# Result: VLM_params get ZERO gradient from action_loss! ❌
```

---

## The Key: Graph Structure, Not Loss Value

### What Doesn't Matter

```python
total_loss = 14.6  # The actual value doesn't determine gradient routing
```

### What Does Matter

The **graph structure** determines which parameters get which gradients:

```python
# This graph structure:
total_loss = f(params_A) + g(params_B)

# Produces these gradients:
∂total_loss/∂params_A = ∂f/∂params_A + 0  # Only from f
∂total_loss/∂params_B = 0 + ∂g/∂params_B  # Only from g
```

Even though `total_loss` is a single number!

---

## Concrete KI Example

### Code

```python
# Forward pass
prefix_out, kv_cache = VLM(prefix_tokens)          # Uses VLM_params
FAST_loss = compute_fast_loss(prefix_out, obs)     # Depends on VLM_params

kv_cache_detached = stop_gradient(kv_cache)        # BARRIER!
suffix_out = ActionExpert(tokens, kv_cache_detached)  # Uses ActionExpert_params
action_loss = compute_action_loss(suffix_out)      # Depends on ActionExpert_params

total_loss = action_loss + FAST_loss               # Single number: 14.6
```

### Graph Dependencies

```python
# What JAX sees internally:
total_loss.dependencies = {
    'action_loss': {
        'params': ActionExpert_params,
        'depends_on_VLM_params': False  # Blocked by stop_gradient!
    },
    'FAST_loss': {
        'params': VLM_params,
        'depends_on_ActionExpert_params': False  # Never connected
    }
}
```

### Gradient Computation

```python
# When you call:
grads = jax.grad(total_loss)(all_params)

# JAX computes:
grads['VLM_params'] = d(total_loss)/d(VLM_params)
                    = d(action_loss)/d(VLM_params) + d(FAST_loss)/d(VLM_params)
                    = 0 (blocked by stop_grad)      + (non-zero from FAST_loss)
                    = only FAST_loss contribution ✓

grads['ActionExpert_params'] = d(total_loss)/d(ActionExpert_params)
                              = d(action_loss)/d(ActionExpert_params) + d(FAST_loss)/d(ActionExpert_params)
                              = (non-zero from action_loss)           + 0 (never connected)
                              = only action_loss contribution ✓
```

---

## Why `stop_gradient` is Critical

Without `stop_gradient`, the graph would be:

```python
# BAD: Without stop_gradient
suffix_out = ActionExpert(tokens, kv_cache)  # Connects to VLM_params!
action_loss = compute_action_loss(suffix_out)

# This creates a path:
action_loss → suffix_out → kv_cache → VLM_params

# Result: VLM_params would get gradients from BOTH losses!
grads['VLM_params'] = FAST_loss_grad + action_loss_grad  # ❌ Wrong!
```

With `stop_gradient`:

```python
# GOOD: With stop_gradient
kv_cache_detached = stop_gradient(kv_cache)
suffix_out = ActionExpert(tokens, kv_cache_detached)
action_loss = compute_action_loss(suffix_out)

# The path is broken:
action_loss → suffix_out → kv_cache_detached → [BARRIER] ❌ VLM_params

# Result: VLM_params only get gradients from FAST_loss!
grads['VLM_params'] = FAST_loss_grad  # ✓ Correct!
```

---

## Visualizing the Magic

### What You See

```python
>>> total_loss
Array(14.6, dtype=float32)  # Just a number
```

### What JAX Sees

```python
>>> total_loss._trace_level  # Internal representation
TraceLevel(
    main_trace=<DynamicJaxprTrace>,
    sublevel=2
)

>>> total_loss._trace.recipe  # Simplified
AddPrimitive(
    x=<Traced<action_loss>>,
    y=<Traced<FAST_loss>>
)
```

JAX maintains this **recipe** (the computational graph) to know how to compute gradients!

---

## Interactive Example

Try this in a Python REPL:

```python
import jax
import jax.numpy as jnp

# Create parameters
params_A = jnp.array(2.0)
params_B = jnp.array(3.0)

def compute_loss(params_A, params_B):
    # Two independent computations
    loss_A = params_A ** 2        # = 4.0
    loss_B = params_B ** 2        # = 9.0
    
    # Combined loss (just a number!)
    total = loss_A + loss_B       # = 13.0
    return total

# Compute gradients
grad_A, grad_B = jax.grad(compute_loss, argnums=(0, 1))(params_A, params_B)

print(f"Total loss: {compute_loss(params_A, params_B)}")  # 13.0
print(f"Gradient w.r.t. params_A: {grad_A}")  # 4.0 (only from loss_A)
print(f"Gradient w.r.t. params_B: {grad_B}")  # 6.0 (only from loss_B)
```

Even though the loss is a single number (13.0), JAX correctly computes:
- `∂total/∂params_A = 4.0` (from `loss_A` only)
- `∂total/∂params_B = 6.0` (from `loss_B` only)

---

## Summary: The "Magic" Explained

| Concept | Explanation |
|---------|-------------|
| **Single Loss Value** | Yes, it's just a number (14.6) |
| **Computational Graph** | JAX tracks *how* that number was computed |
| **Graph Structure** | Determines which parameters connect to which parts of the loss |
| **Backpropagation** | Traverses graph backwards, computing gradients |
| **stop_gradient** | Creates barriers in the graph to isolate gradient flow |
| **Automatic Routing** | JAX automatically routes gradients based on graph structure |

**There's no magic** - it's just the chain rule applied systematically through the computational graph! The graph structure (not the loss value) determines which parameters get which gradients.

---

## Key Takeaway

```python
# This single line:
total_loss = action_loss + FAST_loss

# Creates this graph structure:
#   total_loss
#   /         \
#  /           \
# action_loss   FAST_loss
#     |            |
#     |            |
# ActionExpert   VLM
# params         params

# Which JAX uses to compute these isolated gradients:
# ∂total_loss/∂ActionExpert_params = ∂action_loss/∂ActionExpert_params
# ∂total_loss/∂VLM_params = ∂FAST_loss/∂VLM_params
```

The loss is a number, but the **graph is the map** that tells JAX where gradients should flow!
