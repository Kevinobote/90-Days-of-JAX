# day_015_pmap/main.py

import jax
import jax.numpy as jnp
from jax import random, jit, grad, pmap, vmap, lax
import optax
import time

print("--- Day 15: Introduction to JAX's Parallelism with `jax.pmap` ---")

# --- Part 0: Check Available Devices ---
print(f"\n--- Device Information ---")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")
num_devices = jax.device_count()
print(f"Number of available devices: {num_devices}")
if num_devices < 2:
    print("\nNote: `pmap` benefits from multiple devices (GPUs/TPUs).")
    print("This code will run, but parallelism benefits won't be visible on a single device.")
    print("For single-device setups, Part 2's `pmap` example will be skipped to avoid `num_replicas` error.")
    print("Parts 1 and 3 will still run, demonstrating `pmap`'s structure (Part 1) and full training (Part 3).")


# --- Part 1: Basic `jax.pmap` Example ---
@pmap
def multiply_by_self(x):
    return x * x

print("\n--- Part 1: Basic `jax.pmap` Example ---")
# Ensure the input has a leading dimension matching num_devices
# and enough elements for reshape.
input_data_size = num_devices * 3 if num_devices > 0 else 3 # Ensures at least 3 elements for reshape
input_data = jnp.arange(input_data_size, dtype=jnp.float32).reshape(num_devices, 3)
print(f"Input data for pmap:\n{input_data}")
output_data = multiply_by_self(input_data)
print(f"Output from pmap:\n{output_data}")
print(f"Output shape: {output_data.shape}")


# --- Part 2: `pmap` with Model Parallelism (Replicating Parameters) ---
# THIS SECTION IS MODIFIED TO AVOID THE SINGLE-DEVICE PMAP ERROR
def process_data_with_params_non_pmapped(params, data_batch):
    # This is the original logic, without pmap applied to this function itself.
    # It will be called within a JIT or directly.
    if data_batch.ndim == 1:
        data_batch_2d = jnp.expand_dims(data_batch, axis=0)
    else:
        data_batch_2d = data_batch

    bias_2d = jnp.expand_dims(params['b'], axis=0) if params['b'].ndim == 1 else params['b']
    output = jnp.dot(data_batch_2d, params['w']) + bias_2d
    return output

print("\n--- Part 2: `pmap` with Model Parallelism (Replicating Parameters) ---")

dummy_params = {
    'w': jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
    'b': jnp.array([0.5, 0.5], dtype=jnp.float32)
}

# Dummy data for Part 2. Always shape for a single "batch" on one device.
dummy_data_part2 = jnp.array([[0., 1.]], dtype=jnp.float32) # Shape (1, 2)
print(f"Dummy params: {dummy_params}")
print(f"Dummy data (for single device simulation):\n{dummy_data_part2}")

if num_devices > 1:
    # If multiple devices are available, we can demonstrate pmap with this example
    # First, prepare dummy_data to be sharded across multiple devices
    # Create an input that assumes 2 replicas for this specific example for demonstration
    # (even if num_devices is higher, for clarity of this specific scenario)
    num_replicas_for_demo = 2
    if num_devices < num_replicas_for_demo: # If we have 2+ devices, just use all
        num_replicas_for_demo = num_devices

    dummy_data_sharded = jnp.arange(num_replicas_for_demo * 2, dtype=jnp.float32).reshape(num_replicas_for_demo, 2)

    # Replicate params for pmap
    replicated_dummy_params = jax.tree.map(lambda x: jnp.array([x] * num_replicas_for_demo), dummy_params)

    # Define pmapped version for this specific case (only used if num_devices > 1)
    pmapped_process_data = pmap(process_data_with_params_non_pmapped, in_axes=(None, 0))
    processed_output = pmapped_process_data(replicated_dummy_params, dummy_data_sharded)
    print(f"Processed output (sharded via pmap):\n{processed_output}")
    print(f"Processed output shape: {processed_output.shape}")

else:
    # If only one device, run it without pmap, or with jit on the single device
    print("\nRunning Part 2 example on a single device (without pmap) for conceptual understanding.")
    # To run this with JIT, uncomment the next line and remove the one after it
    # processed_output = jit(process_data_with_params_non_pmapped)(dummy_params, dummy_data_part2)
    processed_output = process_data_with_params_non_pmapped(dummy_params, dummy_data_part2)
    print(f"Processed output (single device):\n{processed_output}")
    print(f"Processed output shape: {processed_output.shape}")

# --- Part 3: Data Parallel Training Example with `pmap` ---
# --- Re-using model and loss from Day 14 ---
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
    return jnp.mean((predictions - y_batch)**2)

# --- The Pmapped Train Step ---
def pmapped_train_step(params, opt_state, X_batch_shard, y_batch_shard):
    loss_value, grads = jax.value_and_grad(mse_loss)(params, X_batch_shard, y_batch_shard)
    grads = lax.pmean(grads, axis_name='devices')
    updates, new_opt_state = optax.adam(0.001).update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_value

pmapped_train_step = pmap(pmapped_train_step, axis_name='devices')

print("\n--- Part 3: Data Parallel Training Example with `pmap` ---")

key = random.PRNGKey(2)
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
replicated_params = jax.tree.map(lambda x: jnp.array([x] * num_devices), init_params_once)
print(f"Replicated params 'w1' shape (num_devices, input_dim, hidden_dim): {replicated_params['w1'].shape}")

optimizer = optax.adam(0.001)
init_opt_state_once = optimizer.init(init_params_once)
replicated_opt_state = jax.tree.map(lambda x: jnp.array([x] * num_devices), init_opt_state_once)
print(f"Replicated opt_state structure: {jax.tree.map(lambda x: x.shape, replicated_opt_state)}")

samples_per_device = num_total_samples // num_devices
sharded_X_train = X_train.reshape(num_devices, samples_per_device, input_dim)
sharded_y_train = y_train.reshape(num_devices, samples_per_device, output_dim)
print(f"Sharded X_train shape (num_devices, samples_per_device, features): {sharded_X_train.shape}")

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

# --- Part 4: Gathering Results (Optional) ---
final_params_on_host = jax.tree.map(lambda x: x[0], replicated_params)
print(f"\nFinal params 'w1' shape (on host, taken from first replica): {final_params_on_host['w1'].shape}")

final_prediction = batched_forward_pass(final_params_on_host, X_train)
final_loss_on_host = jnp.mean((final_prediction - y_train)**2)
print(f"Final loss computed on host with gathered params: {final_loss_on_host:.6f}")