# day_014_practical_nn_training/main.py

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, lax
import optax
import time
from matplotlib import pyplot as plt

print("--- Day 14: Practical Neural Network Training with Optax ---")

# --- Part 1: Data Generation (Revisiting Day 8) ---
key = random.PRNGKey(0)
num_samples = 200
noise_std = 0.8 # Slightly more noisy for a harder learning task

true_w = 2.5
true_b = 1.5

key, subkey = random.split(key)
X = random.uniform(subkey, (num_samples, 1), minval=-10.0, maxval=10.0)
key, subkey = random.split(key)
noise = random.normal(subkey, (num_samples, 1)) * noise_std
y = true_w * X + true_b + noise

print(f"\n--- Data Generation ---")
print(f"X shape: {X.shape}, y shape: {y.shape}")


# --- Part 2: MLP Model Definition (Revisiting Day 8) ---
def init_params(key, input_dim, hidden_dim, output_dim):
    key, w1_key, b1_key, w2_key, b2_key = random.split(key, 5)

    params = {
        'w1': random.normal(w1_key, (input_dim, hidden_dim)) * jnp.sqrt(2.0 / input_dim), # He initialization
        'b1': jnp.zeros((hidden_dim,)),
        'w2': random.normal(w2_key, (hidden_dim, output_dim)) * jnp.sqrt(2.0 / hidden_dim), # He initialization
        'b2': jnp.zeros((output_dim,))
    }
    return params

def forward_pass_single(params, x_single):
    h = jnp.dot(x_single, params['w1']) + params['b1']
    h = jax.nn.relu(h)
    output = jnp.dot(h, params['w2']) + params['b2']
    return output

# Vmap the forward pass for efficient batch processing
# in_axes=(None, 0) means params are broadcasted, x_batch is mapped over its 0-th axis
batched_forward_pass = vmap(forward_pass_single, in_axes=(None, 0))

print("\n--- MLP Model Definition ---")
input_dim = X.shape[1]
hidden_dim = 64 # Larger hidden layer
output_dim = y.shape[1]
model_key = random.PRNGKey(1)
initial_params = init_params(model_key, input_dim, hidden_dim, output_dim)
print(f"Initial params structure: {jax.tree.map(lambda x: x.shape, initial_params)}")


# --- Part 3: Loss Function ---
def mse_loss(params, X_batch, y_batch):
    predictions = batched_forward_pass(params, X_batch)
    return jnp.mean((predictions - y_batch)**2)

print(f"\nInitial loss: {mse_loss(initial_params, X, y):.4f}")


# --- Part 4: Optimizer and JIT-compiled Train Step ---
learning_rate = 0.005 # Slightly lower learning rate
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(initial_params)

@jit
def train_step(params, opt_state, X_batch, y_batch):
    # Calculate loss and gradients
    loss_value, grads = jax.value_and_grad(mse_loss)(params, X_batch, y_batch)

    # Apply updates using Optax
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss_value

print("\n--- Optimizer and JIT-compiled Train Step ---")
print(f"Optimizer: {optimizer}")


# --- Part 5: Training Loop with `jax.lax.fori_loop` for Epochs ---
# We'll use fori_loop for the epochs, as the number of epochs is fixed.
# For batching *within* an epoch, we'll keep it simple for now (full batch)
# or you could nest another fori_loop/scan for mini-batches.

num_epochs = 1000
batch_size = num_samples # Using full batch for simplicity

# The `train_epoch` function will encapsulate one full pass over the data
@jit # JIT this outer loop (or the whole fori_loop)
def train_model(initial_params, initial_opt_state, X_data, y_data, num_epochs):
    # Define the loop body for lax.fori_loop
    def epoch_body_fun(epoch_idx, carry):
        params, opt_state, losses_list_jnp = carry

        # For simplicity, we use the full dataset as a single batch here.
        # In a real scenario, you'd shuffle and iterate through mini-batches.
        X_batch, y_batch = X_data, y_data

        new_params, new_opt_state, current_loss = train_step(params, opt_state, X_batch, y_batch)

        # Store the loss. We append it to a growing array.
        # This is not memory efficient for very many epochs as it involves reallocations.
        # For real logging, you'd collect Python list outside JIT and convert.
        # For this example, we'll collect only the last loss, or a fixed-size array
        # Or, we can simply return the last loss and print it.
        # Let's collect a fixed number of losses for demonstration.
        # Or, for simplicity and to avoid complex JAX array growth patterns,
        # we just log the *final* loss from the loop.
        # Let's just track the last loss in the carry for demonstration.

        return (new_params, new_opt_state, current_loss) # carry for next iteration

    # Initial carry for the fori_loop: (params, opt_state, initial_loss)
    # We start with params and opt_state, and a dummy loss for the first iteration.
    # The initial_loss will be replaced immediately.
    initial_carry = (initial_params, initial_opt_state, mse_loss(initial_params, X_data, y_data))

    final_params, final_opt_state, final_loss = lax.fori_loop(
        0, num_epochs, epoch_body_fun, initial_carry
    )
    return final_params, final_opt_state, final_loss

print(f"\n--- Training Loop ({num_epochs} epochs) with `jax.lax.fori_loop` ---")
start_time = time.time()
final_params, final_opt_state, final_training_loss = train_model(initial_params, opt_state, X, y, num_epochs)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.4f}s")
print(f"Final training loss: {final_training_loss:.4f}")

# Note: To log losses per epoch, you would typically collect them *outside* the jitted
# train_model function using a Python list and append `current_loss.item()`
# after calling `train_step`, or use `lax.scan` for accumulation if it's feasible
# within the JIT, as shown in Day 13.
# For this example, we just get the final loss.


# --- Part 6: Visualization ---
print("\n--- Visualization ---")
plt.figure(figsize=(10, 5))

# Plotting losses would require collecting them from inside the loop,
# which is more complex with `fori_loop` inside `jit`.
# For now, we'll just plot the data and prediction.

plt.scatter(X, y, label='True Data', alpha=0.7)
X_test = jnp.linspace(jnp.min(X) - 1, jnp.max(X) + 1, 200).reshape(-1, 1) # Extend range slightly
y_pred = batched_forward_pass(final_params, X_test)
plt.plot(X_test, y_pred, color='red', label='Trained Model Prediction')
plt.title('Data and Trained Model Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\nPractical training loop demonstrating composition of JAX transforms complete!")