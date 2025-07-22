# day_008_basic_nn/main.py

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap # vmap isn't strictly used today, but good to keep for completeness
import optax
import time
from matplotlib import pyplot as plt

print("--- Day 8: Basic Neural Network (MLP) in JAX ---")

# --- Part 1: Data Generation ---
key = random.PRNGKey(0)
num_samples = 100
noise_std = 0.5

true_w = 2.0
true_b = 1.0

key, subkey = random.split(key)
X = random.uniform(subkey, (num_samples, 1), minval=-5.0, maxval=5.0)
key, subkey = random.split(key)
noise = random.normal(subkey, (num_samples, 1)) * noise_std
y = true_w * X + true_b + noise

print(f"\n--- Data Generation ---")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"First 5 X values:\n{X[:5].flatten()}")
print(f"First 5 y values:\n{y[:5].flatten()}")


# --- Part 2: MLP Model Definition ---
def init_params(key, input_dim, hidden_dim, output_dim):
    key, w1_key, b1_key, w2_key, b2_key = random.split(key, 5)

    params = {
        'w1': random.normal(w1_key, (input_dim, hidden_dim)) * 0.01,
        'b1': jnp.zeros((hidden_dim,)),
        'w2': random.normal(w2_key, (hidden_dim, output_dim)) * 0.01,
        'b2': jnp.zeros((output_dim,))
    }
    return params

def forward_pass(params, x):
    h = jnp.dot(x, params['w1']) + params['b1']
    h = jax.nn.relu(h)
    output = jnp.dot(h, params['w2']) + params['b2']
    return output

print("\n--- MLP Model Definition ---")
input_dim = 1
hidden_dim = 10
output_dim = 1
model_key = random.PRNGKey(1)
initial_params = init_params(model_key, input_dim, hidden_dim, output_dim)
# CORRECTED LINE:
print(f"Initial params structure: {jax.tree.map(lambda x: x.shape, initial_params)}")
# JAX's tree_map was moved. jax.tree.map is the current preferred.


# --- Part 3: Loss Function ---
def loss_fn(params, X_batch, y_batch):
    predictions = forward_pass(params, X_batch)
    return jnp.mean((predictions - y_batch)**2)

print(f"\nInitial loss: {loss_fn(initial_params, X, y):.4f}")


# --- Part 4: Optimization with Optax ---
learning_rate = 0.01
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(initial_params)

@jit
def update_step(params, opt_state, X_batch, y_batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, X_batch, y_batch)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss, grads

print("\n--- Optimization with Optax ---")
print(f"Optimizer: {optimizer}")


# --- Part 5: Training Loop ---
num_epochs = 1000
batch_size = 32

params = initial_params
losses = []

print(f"\n--- Training Loop ({num_epochs} epochs) ---")
for epoch in range(num_epochs):
    X_batch, y_batch = X, y

    params, opt_state, loss, grads = update_step(params, opt_state, X_batch, y_batch)
    losses.append(loss)

    if epoch % 100 == 0 or epoch == num_epochs - 1:
        grad_norm = jnp.sqrt(sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads)))
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Grad Norm: {grad_norm:.4f}")

print(f"Final loss: {losses[-1]:.4f}")
# Also corrected this line just in case, though tree_map might be a bit more robust here
# as it's printing all params, not just shapes.
print(f"Trained parameters: {jax.tree.map(lambda x: x.flatten(), params)}")


# --- Part 6: Visualization ---
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')

plt.subplot(1, 2, 2)
plt.scatter(X, y, label='True Data', alpha=0.7)
X_test = jnp.linspace(jnp.min(X), jnp.max(X), 100).reshape(-1, 1)
y_pred = forward_pass(params, X_test)
plt.plot(X_test, y_pred, color='red', label='Trained Model Prediction')
plt.title('Data and Model Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()