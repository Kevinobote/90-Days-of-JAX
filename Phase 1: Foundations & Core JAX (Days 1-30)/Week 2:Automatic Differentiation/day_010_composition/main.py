# day_010_composition/main.py

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import time
import optax # For a quick optimizer example

print("--- Day 10: Combining All JAX Transformations and Next Steps ---")

# --- Part 1: Revisiting a JIT-compiled Gradient ---
# This is the most common combination for ML training loops.

# Function definition (simple squared difference)
def squared_diff(x, y):
    return jnp.sum((x - y)**2)

# Combined JIT and Grad
# `jax.jit(jax.grad(func))` is idiomatic: compute gradient, then compile the gradient computation.
jit_grad_squared_diff = jit(grad(squared_diff))

print("\n--- JIT-compiled Gradient (`jit(grad(func))`) ---")
val_x = jnp.array(5.0)
val_y = jnp.array(3.0)

start_time = time.time()
grad_result_first = jit_grad_squared_diff(val_x, val_y).block_until_ready()
end_time = time.time()
print(f"Gradient (first run, includes compilation): {grad_result_first}, Time: {end_time - start_time:.6f}s")
# Expected: 2*(x-y) = 2*(5-3) = 4.0

start_time = time.time()
grad_result_subsequent = jit_grad_squared_diff(val_x, jnp.array(10.0)).block_until_ready() # Change y
end_time = time.time()
print(f"Gradient (subsequent run, different data): {grad_result_subsequent}, Time: {end_time - start_time:.6f}s")
# Expected: 2*(5-10) = -10.0


# --- Part 2: JIT-compiled Vmapped Function ---
# Efficiently apply a function over batches of data.

def elementwise_pow(x, power):
    return x**power

# vmap over the first argument (x), broadcast the second (power)
vmapped_pow = vmap(elementwise_pow, in_axes=(0, None))
jit_vmapped_pow = jit(vmapped_pow)

print("\n--- JIT-compiled Vmapped Function (`jit(vmap(func))`) ---")
batch_x = jnp.array([1.0, 2.0, 3.0, 4.0])
fixed_power = jnp.array(2.0)

start_time = time.time()
vmap_result_first = jit_vmapped_pow(batch_x, fixed_power).block_until_ready()
end_time = time.time()
print(f"Vmap result (first run): {vmap_result_first}, Time: {end_time - start_time:.6f}s")
# Expected: [1, 4, 9, 16]

# Change batch_x data
new_batch_x = jnp.array([5.0, 6.0, 7.0, 8.0])
start_time = time.time()
vmap_result_subsequent = jit_vmapped_pow(new_batch_x, fixed_power).block_until_ready()
end_time = time.time()
print(f"Vmap result (subsequent run): {vmap_result_subsequent}, Time: {end_time - start_time:.6f}s")
# Expected: [25, 36, 49, 64]


# --- Part 3: Combining Grad and Vmap for Batch Gradients ---
# Very common in ML: compute gradients for a whole batch simultaneously.

# Assume a simple "loss per example" function
def loss_per_example(weights, feature, target):
    prediction = weights @ feature # Simple linear model
    return (prediction - target)**2 # Squared error for one example

# We want to calculate the gradient of the total batch loss w.r.t. weights.
# Approach:
# 1. vmap the `loss_per_example` function over the `feature` and `target` axes (inputs 1 and 2).
#    This gives us `batch_loss_fn(weights, batch_features, batch_targets)`.
#    The output of `batch_loss_fn` will be a vector of losses.
# 2. Sum the output of the vmapped function to get a single scalar batch loss.
# 3. Grad this summed batch loss with respect to `weights` (input 0).
# 4. JIT the entire thing for performance.

# 1. Vmap `loss_per_example`
# weights (None - broadcast), feature (0 - batch dim), target (0 - batch dim)
batched_loss_per_example = vmap(loss_per_example, in_axes=(None, 0, 0))

# 2. Sum the batched loss to get a scalar
def total_batch_loss(weights, features, targets):
    individual_losses = batched_loss_per_example(weights, features, targets)
    return jnp.sum(individual_losses)

# 3. Grad the total batch loss wrt weights (first argument)
grad_total_batch_loss = grad(total_batch_loss, argnums=0) # argnums=0 for weights

# 4. JIT the final gradient function
jit_grad_total_batch_loss = jit(grad_total_batch_loss)

print("\n--- JIT-compiled Vmapped Gradient (`jit(grad(vmap(func)))`) ---")

# Generate dummy data for this example
key = random.PRNGKey(10)
num_batch_samples = 128
feature_dim = 5

key, subkey1, subkey2, subkey3 = random.split(key, 4)
initial_weights = random.normal(subkey1, (feature_dim,)) * 0.1
batch_features = random.normal(subkey2, (num_batch_samples, feature_dim))
batch_targets = random.normal(subkey3, (num_batch_samples,)) * 5.0 # Random targets

start_time = time.time()
batch_grads_first = jit_grad_total_batch_loss(initial_weights, batch_features, batch_targets).block_until_ready()
end_time = time.time()
print(f"Batch Gradients (first run): {batch_grads_first}, Time: {end_time - start_time:.6f}s")

# Subsequent run with different data
key, subkey2, subkey3 = random.split(key, 3)
new_batch_features = random.normal(subkey2, (num_batch_samples, feature_dim))
new_batch_targets = random.normal(subkey3, (num_batch_samples,)) * 5.0

start_time = time.time()
batch_grads_subsequent = jit_grad_total_batch_loss(initial_weights, new_batch_features, new_batch_targets).block_until_ready()
end_time = time.time()
print(f"Batch Gradients (subsequent run): {batch_grads_subsequent}, Time: {end_time - start_time:.6f}s")


# --- Part 4: A simplified training loop structure ---
# This is how you'd typically set up an actual ML training loop.

# Model parameters (using PyTree from Day 8)
def init_params_nn(key, input_dim, hidden_dim, output_dim):
    key, w1_key, b1_key, w2_key, b2_key = random.split(key, 5)
    params = {
        'w1': random.normal(w1_key, (input_dim, hidden_dim)) * 0.01,
        'b1': jnp.zeros((hidden_dim,)),
        'w2': random.normal(w2_key, (hidden_dim, output_dim)) * 0.01,
        'b2': jnp.zeros((output_dim,))
    }
    return params

# Single example forward pass
def forward_pass_single(params, x_single):
    h = jnp.dot(x_single, params['w1']) + params['b1']
    h = jax.nn.relu(h)
    output = jnp.dot(h, params['w2']) + params['b2']
    return output

# Batch forward pass using vmap
# We apply the single example forward_pass to a batch of inputs X.
# The 'params' are broadcast (None), 'x_batch' is mapped over its 0-th axis.
batched_forward_pass = vmap(forward_pass_single, in_axes=(None, 0))

# Loss function for a batch (returns scalar)
def batch_loss_nn(params, X_batch, y_batch):
    predictions = batched_forward_pass(params, X_batch)
    return jnp.mean((predictions - y_batch)**2) # Mean Squared Error over the batch

# Combined JIT and Grad for the training step
# This function returns loss and gradients w.r.t. params.
@jit
def train_step(params, opt_state, X_batch, y_batch):
    loss, grads = jax.value_and_grad(batch_loss_nn)(params, X_batch, y_batch)
    updates, new_opt_state = optax.adam(0.01).update(grads, opt_state, params) # Re-initialize optimizer each step for simplicity in this example
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

print("\n--- Simplified Training Loop (Combining all transforms) ---")

input_dim_nn = 1
hidden_dim_nn = 10
output_dim_nn = 1
model_key_nn = random.PRNGKey(20)
params_nn = init_params_nn(model_key_nn, input_dim_nn, hidden_dim_nn, output_dim_nn)
optimizer_nn = optax.adam(0.01) # Define once outside the loop
opt_state_nn = optimizer_nn.init(params_nn)

# Dummy data for the loop example
key_data, _ = random.split(random.PRNGKey(21))
X_train_dummy = random.normal(key_data, (100, input_dim_nn))
y_train_dummy = random.normal(key_data, (100, output_dim_nn))

num_epochs_nn = 50
print(f"Training for {num_epochs_nn} epochs...")
for epoch in range(num_epochs_nn):
    params_nn, opt_state_nn, current_loss = train_step(params_nn, opt_state_nn, X_train_dummy, y_train_dummy)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}, Loss: {current_loss:.4f}")

print("Training loop complete!")


# --- Part 5: Next Steps in JAX ---

print("\n--- What's Next in Your JAX Journey? ---")
print("You've covered the fundamental transformations:")
print("1. `jax.Array`: Immutable arrays, like NumPy arrays but on accelerators.")
print("2. `jax.jit`: Just-in-Time compilation for speed.")
print("3. `jax.grad`: Automatic differentiation (reverse-mode AD for scalar outputs).")
print("4. `jax.random`: Functional pseudo-random number generation.")
print("5. `jax.vmap`: Automatic vectorization for batching operations.")
print("6. `jax.jvp` & `jax.vjp`: The underlying forward/reverse mode AD primitives.")
print("\nTo go further, explore these key areas:")
print("- **State Management (e.g., Flax, Equinox):** How to manage mutable state (like model parameters) in JAX's functional paradigm. Libraries like Flax and Equinox provide powerful abstractions.")
print("- **Custom Gradients:** For complex operations where JAX's default gradient might not be what you need, or for performance. `jax.custom_jvp` and `jax.custom_vjp`.")
print("- **Control Flow:** `jax.lax.cond`, `jax.lax.while_loop`, `jax.lax.fori_loop` for JIT-compatible control flow.")
print("- **Parallelism (`jax.pmap`):** Running computations across multiple devices (e.g., multiple GPUs, TPUs).")
print("- **Tree Utilities (`jax.tree_util` or `jax.tree`):** Powerful tools for manipulating nested data structures (PyTrees).")
print("- **Debugging:** Tools like `jax.debug.print` for inspecting values inside JITted functions.")
print("- **JIT Cache and AOT Compilation:** Understanding how JAX caches compiled functions and advanced compilation options.")
print("\nCongratulations on completing the JAX Foundation Core challenge!")
print("You now have a solid understanding of JAX's core principles and are well-equipped to dive into more advanced topics and build powerful machine learning models.")