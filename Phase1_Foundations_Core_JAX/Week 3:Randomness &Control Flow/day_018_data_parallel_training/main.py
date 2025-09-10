# day_018_data_parallel_training/main.py

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.lax import pmean
import optax
import time

print("--- Day 18: Building a Complete Data-Parallel Training Loop ---")

# --- Part 0: Device Information ---
print(f"\n--- Device Information ---")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")
num_devices = jax.device_count()
print(f"Number of available devices: {num_devices}")

if num_devices < 2:
    print("\nNote: `pmap` benefits from multiple devices (GPUs/TPUs).")
    print("This training loop will run, but parallelism benefits won't be visible.")
    print("On a single device, `pmap` simulates a single replica, and `pmean` acts as an identity function.")


# --- Part 1: Model Definition ---
def init_params(key, input_dim, hidden_dim, output_dim):
    key, w1_key, b1_key, w2_key, b2_key = random.split(key, 5)
    params = {
        'w1': random.normal(w1_key, (input_dim, hidden_dim)) * jnp.sqrt(2.0 / input_dim),
        'b1': jnp.zeros((hidden_dim,)),
        'w2': random.normal(w2_key, (hidden_dim, output_dim)) * jnp.sqrt(2.0 / hidden_dim),
        'b2': jnp.zeros((output_dim,))
    }
    return params

def forward_pass_single(params, x_single):
    h = jnp.dot(x_single, params['w1']) + params['b1']
    h = jax.nn.relu(h)
    output = jnp.dot(h, params['w2']) + params['b2']
    return output

batched_forward_pass = vmap(forward_pass_single, in_axes=(None, 0))

def mse_loss(params, X_batch, y_batch):
    predictions = batched_forward_pass(params, X_batch)
    return jnp.mean((predictions - y_batch) ** 2)


# --- Part 2: The Data-Parallel Training Step ---
def train_step(params, opt_state, X_batch_shard, y_batch_shard):
    loss_value, grads = jax.value_and_grad(mse_loss)(params, X_batch_shard, y_batch_shard)
    grads = pmean(grads, axis_name='devices')
    updates, new_opt_state = optax.adam(0.001).update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_value

# Wrap with pmap
pmapped_train_step = jax.pmap(train_step, axis_name="devices")


# --- Part 3: Main Training Loop ---
print("\n--- Part 3: Main Training Loop ---")

key = random.PRNGKey(42)
num_total_samples = 256
if num_total_samples % num_devices != 0:
    num_total_samples = (num_total_samples // num_devices + 1) * num_devices

input_dim = 1
output_dim = 1
hidden_dim = 32

key, subkey_x, subkey_noise = random.split(key, 3)
X_train = random.uniform(subkey_x, (num_total_samples, input_dim), minval=-5.0, maxval=5.0)
y_train = 2.0 * X_train + 1.0 + random.normal(subkey_noise, (num_total_samples, output_dim)) * 0.5

master_key = random.PRNGKey(3)
init_params_once = init_params(master_key, input_dim, hidden_dim, output_dim)
optimizer = optax.adam(0.001)
init_opt_state_once = optimizer.init(init_params_once)

replicated_params = jax.tree.map(lambda x: jnp.array([x] * num_devices), init_params_once)
replicated_opt_state = jax.tree.map(lambda x: jnp.array([x] * num_devices), init_opt_state_once)
print(f"Replicated params 'w1' shape: {replicated_params['w1'].shape}")
print(f"Replicated opt_state structure: {jax.tree.map(lambda x: x.shape, replicated_opt_state)}")

samples_per_device = num_total_samples // num_devices
sharded_X_train = X_train.reshape(num_devices, samples_per_device, input_dim)
sharded_y_train = y_train.reshape(num_devices, samples_per_device, output_dim)
print(f"Sharded X_train shape: {sharded_X_train.shape}")

num_training_steps = 1000
print(f"\nTraining for {num_training_steps} steps...")
start_time = time.time()
for step in range(num_training_steps):
    replicated_params, replicated_opt_state, per_device_loss = pmapped_train_step(
        replicated_params, replicated_opt_state, sharded_X_train, sharded_y_train
    )
    current_mean_loss = jnp.mean(per_device_loss)
    if step % 100 == 0 or step == num_training_steps - 1:
        print(f"Step {step:4d}, Loss: {current_mean_loss:.6f}")

end_time = time.time()
print(f"\nPmap training completed in {end_time - start_time:.4f}s")
print(f"Final mean loss: {current_mean_loss:.6f}")

# --- Part 4: Evaluation ---
final_params_on_host = jax.tree.map(lambda x: x[0], replicated_params)
print(f"\nFinal params 'w1' shape (on host, from first replica): {final_params_on_host['w1'].shape}")

final_prediction = batched_forward_pass(final_params_on_host, X_train)
final_loss_on_host = jnp.mean((final_prediction - y_train) ** 2)
print(f"Final loss computed on host with gathered params: {final_loss_on_host:.6f}")

print("\n--- Day 18 Conclusion ---")
print("You've successfully built a complete data-parallel training loop in JAX!")
print("This pattern of sharding data, replicating state, and using `pmean` for gradient aggregation")
print("is the foundation for scalable distributed training in JAX, as used by libraries like Flax and Orbax.")
