# day_005_jit_and_grad/main.py

import jax
import jax.numpy as jnp
import time

print("--- Day 5: Combining `jax.jit` and `jax.grad` ---")

# --- Part 1: JITting a function that calculates a scalar loss ---

# Define a simple loss function (e.g., mean squared error for a single predicted value)
def simple_loss(prediction, target):
    return jnp.mean((prediction - target)**2)

# Compile the loss function with JIT
jit_simple_loss = jax.jit(simple_loss)

print("\n--- JITting a simple loss function ---")
pred = jnp.array(5.0)
tgt = jnp.array(7.0)

start_time = time.time()
loss_val_nojit = simple_loss(pred, tgt).block_until_ready()
end_time = time.time()
print(f"Loss (No JIT): {loss_val_nojit:.4f}, Time: {end_time - start_time:.6f} seconds")

start_time = time.time()
loss_val_jit_first = jit_simple_loss(pred, tgt).block_until_ready()
end_time = time.time()
print(f"Loss (JIT First Run): {loss_val_jit_first:.4f}, Time: {end_time - start_time:.6f} seconds (includes compilation)")

start_time = time.time()
loss_val_jit_subsequent = jit_simple_loss(pred, tgt).block_until_ready()
end_time = time.time()
print(f"Loss (JIT Subsequent Run): {loss_val_jit_subsequent:.4f}, Time: {end_time - start_time:.6f} seconds (compiled)")


# --- Part 2: Grad of a JITted function ---

# We want to find the gradient of the simple_loss with respect to 'prediction'
# The function passed to jax.grad must return a scalar. Our `simple_loss` does.

# Option A: Apply grad to the original function, then jit the result
grad_loss_nojit = jax.grad(simple_loss)
jit_grad_loss_A = jax.jit(grad_loss_nojit)

print("\n--- Grad of a JITted function (Option A: grad then jit) ---")
start_time = time.time()
grad_val_A_first = jit_grad_loss_A(pred, tgt).block_until_ready()
end_time = time.time()
print(f"Grad (A, JIT First Run): {grad_val_A_first:.4f}, Time: {end_time - start_time:.6f} seconds")

start_time = time.time()
grad_val_A_subsequent = jit_grad_loss_A(pred, tgt).block_until_ready()
end_time = time.time()
print(f"Grad (A, JIT Subsequent Run): {grad_val_A_subsequent:.4f}, Time: {end_time - start_time:.6f} seconds")


# Option B: Apply jit to the original function, then grad the result
# This is often the more natural way in JAX.
# When you grad a jit-compiled function, JAX automatically traces and compiles the gradient computation.
grad_jit_loss_B = jax.grad(jit_simple_loss)

print("\n--- Grad of a JITted function (Option B: jit then grad) ---")
start_time = time.time()
grad_val_B_first = grad_jit_loss_B(pred, tgt).block_until_ready()
end_time = time.time()
print(f"Grad (B, JIT First Run): {grad_val_B_first:.4f}, Time: {end_time - start_time:.6f} seconds")

start_time = time.time()
grad_val_B_subsequent = grad_jit_loss_B(pred, tgt).block_until_ready()
end_time = time.time()
print(f"Grad (B, JIT Subsequent Run): {grad_val_B_subsequent:.4f}, Time: {end_time - start_time:.6f} seconds")

# Verify that both options yield the same result
print(f"Are results from Option A and B close? {jnp.allclose(grad_val_A_subsequent, grad_val_B_subsequent)}")

# --- Part 3: A more realistic example: Gradient for Linear Regression ---

# Define a simple linear model: y_pred = w * x + b
def linear_model(params, x):
    w, b = params
    return w * x + b

# Define the loss function (Mean Squared Error)
def mse_loss(params, x, y_true):
    y_pred = linear_model(params, x)
    return jnp.mean((y_pred - y_true)**2)

# Generate some dummy data
key = jax.random.PRNGKey(0) # We'll learn about PRNGKeys later, for now just use it
true_w = 2.0
true_b = 1.0
num_samples = 100
x_data = jax.random.uniform(key, (num_samples,), minval=0.0, maxval=10.0)
noise = jax.random.normal(key, (num_samples,)) * 0.5
y_data = true_w * x_data + true_b + noise

# Initial parameters
initial_params = jnp.array([0.0, 0.0]) # [initial_w, initial_b]

# Get the JIT-compiled gradient function for the loss
# We want gradients w.r.t. `params` (the first argument, so argnums=0 is default or explicit)
# x_data and y_true are not differentiated, so they are just passed through
grad_mse = jax.jit(jax.grad(mse_loss, argnums=0)) # Or simply jax.jit(jax.grad(mse_loss))

print("\n--- Linear Regression Gradient Example ---")
print(f"Initial parameters: {initial_params}")
initial_loss = mse_loss(initial_params, x_data, y_data)
print(f"Initial loss: {initial_loss:.4f}")

# Compute gradients for initial parameters
start_time = time.time()
gradients = grad_mse(initial_params, x_data, y_data).block_until_ready()
end_time = time.time()
print(f"Gradients at initial params: {gradients}")
print(f"Time to compute gradients: {end_time - start_time:.6f} seconds (includes compilation)")

# Subsequent call (should be much faster)
start_time = time.time()
gradients_subsequent = grad_mse(initial_params, x_data, y_data).block_until_ready()
end_time = time.time()
print(f"Subsequent time to compute gradients: {end_time - start_time:.6f} seconds (compiled)")