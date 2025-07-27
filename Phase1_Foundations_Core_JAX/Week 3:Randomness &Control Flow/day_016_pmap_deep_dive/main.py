# day_016_pmap_deep_dive/main.py

import jax
import jax.numpy as jnp
from jax import random, jit, grad, pmap, vmap
from jax.lax import pmean # Import pmean specifically
import time

print("--- Day 16: Deep Dive into `jax.pmap` - Control and Collective Operations ---")

# --- Part 0: Device Information ---
print(f"\n--- Device Information ---")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")
num_devices = jax.device_count()
print(f"Number of available devices: {num_devices}")

if num_devices < 2:
    print("\nNote: `pmap` benefits from multiple devices (GPUs/TPUs).")
    print("This code will run, but parallelism benefits won't be visible on a single device.")
    print("For single-device setups, some `pmap` examples might behave conceptually but not leverage true parallelism.")
    print("We'll use `jit` or direct execution for certain examples to avoid `pmap`'s single-device simulation quirks.")

# --- Helper for conditionally running pmap or jit ---
def run_on_device(func, *args, **kwargs):
    if num_devices > 1:
        # If multiple devices, apply pmap
        pmapped_func = pmap(func, **kwargs)
        # Ensure arguments are correctly sharded/replicated for pmap
        # This simple helper assumes in_axes=(0, 0, ...) by default for demonstration
        # For more complex in_axes, this helper would need more logic or be passed explicitly
        return pmapped_func(*args)
    else:
        # If single device, apply jit for performance or run directly
        print(f"(Running {func.__name__} on single device, using jit)")
        # For simplicity in this example, we will manually handle input shapes for jit
        # The main purpose is to prevent pmap errors when num_devices=1
        jitted_func = jit(func)
        # We assume for single-device simulation, inputs are usually not sharded (leading dim removed)
        # If input was (1, ...), we take its 0-th element to simulate the "per-device" input
        processed_args = [arg[0] if (isinstance(arg, jnp.ndarray) and arg.shape[0] == 1 and arg.ndim > 1) else arg for arg in args]
        return jitted_func(*processed_args)


# --- Part 1: Controlling Data Flow with `in_axes` and `out_axes` ---
print("\n--- Part 1: Controlling Data Flow with `in_axes` and `out_axes` ---")

# @pmap'd function to demonstrate in_axes and out_axes
def process_data_flow(x, y, multiplier):
    # x is sharded (in_axes=0 by default)
    # y is replicated (in_axes=None)
    # multiplier is static (in_axes=-1 or static_broadcasted_argnums)

    local_result = (x + y) * multiplier
    return local_result, jnp.sum(local_result) # Return local result and its sum

# Example 1.1: Default in_axes=(0, 0, ...)
# `multiplier` will be treated as sharded by default, which is wrong.
# We need to specify in_axes for all arguments.
print("\nExample 1.1: in_axes=(0, None, None) - Shard x, Replicate y, Replicate multiplier")
# Data for pmap:
# x_data will be sharded
# y_data will be replicated
# multiplier will be broadcasted (replicated)
x_input = jnp.arange(num_devices * 3, dtype=jnp.float32).reshape(num_devices, 3) # (num_devices, 3)
y_input = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32) # (3,) - will be replicated
multiplier_val = 2.0 # scalar - will be replicated

if num_devices > 1:
    pmapped_process_data_flow = pmap(process_data_flow, in_axes=(0, None, None))
    # Note: If y_input was (num_devices, 3), and you wanted to replicate each of those, you'd need a jax.tree.map to wrap it.
    # Here, y_input (3,) is simply broadcasted to all devices.
    # multiplier_val (scalar) is also broadcasted.
    result_sharded, result_sum_sharded = pmapped_process_data_flow(x_input, y_input, multiplier_val)
    print(f"x_input (sharded):\n{x_input}")
    print(f"y_input (replicated original):\n{y_input}")
    print(f"multiplier_val (replicated original): {multiplier_val}")
    print(f"Result (sharded local_result):\n{result_sharded}")
    print(f"Result (sharded local_sum):\n{result_sum_sharded}")
    print(f"Result shapes: {result_sharded.shape}, {result_sum_sharded.shape}")
else:
    print("(Skipping pmap example 1.1 - running direct/jit for single device)")
    # For single device, x_input is [[0., 1., 2.]]. We process the 0-th element as input to simulate.
    x_single = x_input[0]
    result_sharded_sim, result_sum_sharded_sim = jit(process_data_flow)(x_single, y_input, multiplier_val)
    print(f"x_input (simulated sharded input for 0th device):\n{x_single}")
    print(f"y_input (replicated original):\n{y_input}")
    print(f"multiplier_val (replicated original): {multiplier_val}")
    print(f"Result (simulated local_result):\n{result_sharded_sim}")
    print(f"Result (simulated local_sum):\n{result_sum_sharded_sim}")
    print(f"Result shapes: {result_sharded_sim.shape}, {result_sum_sharded_sim.shape}")


# Example 1.2: out_axes=(0, None) - Shard first output, Replicate second output
# Let's say we want to return the local result (sharded) and the overall average (replicated)
def compute_and_replicate_average(x_local):
    local_sum = jnp.sum(x_local)
    # This requires a collective operation if we want a true average across all devices
    # For now, let's just simulate returning something that *would* be replicated
    # We'll properly use pmean in Part 2.
    dummy_replicated_value = jnp.array(42.0) # This will technically be local still
    return x_local, dummy_replicated_value

print("\nExample 1.2: out_axes=(0, None) - Shard first output, Replicate second output (conceptual)")
x_input_2 = jnp.arange(num_devices * 2, dtype=jnp.float32).reshape(num_devices, 2) # (num_devices, 2)

if num_devices > 1:
    pmapped_compute_and_replicate = pmap(compute_and_replicate_average, in_axes=0, out_axes=(0, None))
    # out_axes=(0, None) means the first return value will retain its mapped axis (0),
    # and the second return value will have its mapped axis removed (None), effectively replicating it.
    output_sharded, output_replicated = pmapped_compute_and_replicate(x_input_2)
    print(f"Input x_input_2:\n{x_input_2}")
    print(f"Output sharded (local results):\n{output_sharded}")
    print(f"Output replicated (dummy value):\n{output_replicated}")
    print(f"Output shapes: {output_sharded.shape}, {output_replicated.shape}")
else:
    print("(Skipping pmap example 1.2 - running direct/jit for single device)")
    x_single_2 = x_input_2[0]
    output_sharded_sim, output_replicated_sim = jit(compute_and_replicate_average)(x_single_2)
    print(f"Input x_single_2 (simulated sharded input):\n{x_single_2}")
    print(f"Output sharded (local results):\n{output_sharded_sim}")
    print(f"Output replicated (dummy value):\n{output_replicated_sim}")
    print(f"Output shapes: {output_sharded_sim.shape}, {output_replicated_sim.shape}")


# --- Part 2: Collective Operations with `axis_name` and `jax.lax.pmean` ---
print("\n--- Part 2: Collective Operations with `axis_name` and `jax.lax.pmean` ---")

# `axis_name` provides a handle to the mapped axis within the pmap'd function,
# allowing collective operations (like pmean, psum, pmax, pmin, pscatter, ppermute etc.)
# to communicate across devices along that specific axis.

def compute_average_across_devices(local_data_shard):
    # local_data_shard is the slice of data for this device
    # Calculate the sum of elements on this device
    local_sum = jnp.sum(local_data_shard)

    # Use pmean to average the local_sum across all devices along the 'batch' axis
    # The result of pmean will be the same on all devices.
    global_average_of_sums = pmean(local_sum, axis_name='batch')

    # Also compute global average of the *original* sharded data elements
    global_data_average = pmean(local_data_shard, axis_name='batch')

    return local_sum, global_average_of_sums, global_data_average

def compute_average_single_device(local_data_shard):
    # Single device version without pmean
    local_sum = jnp.sum(local_data_shard)
    # For single device, global average is just the local value
    global_average_of_sums = local_sum
    global_data_average = local_data_shard
    return local_sum, global_average_of_sums, global_data_average

# Input data to be sharded
data_for_pmean = jnp.array([[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0],
                            [7.0, 8.0]], dtype=jnp.float32)

# Adjust input data based on num_devices
if num_devices == 1:
    # If only one device, we can't truly demonstrate pmean.
    # Simulate a single shard being processed.
    # The pmean will effectively just return the local value.
    data_for_pmean_sharded = data_for_pmean[:1] # Take only the first row
    print("\n(Running pmean example on single device - pmean will return local value)")
elif num_devices > 1 and num_devices < data_for_pmean.shape[0]:
    # If we have multiple devices but less than the data rows,
    # we take a subset of data rows to match num_devices.
    data_for_pmean_sharded = data_for_pmean[:num_devices]
    print(f"\n(Running pmean example on {num_devices} devices, using first {num_devices} rows of data)")
elif num_devices > data_for_pmean.shape[0]:
    # If more devices than data, repeat data to fit devices for a better demo
    data_for_pmean_sharded = jnp.tile(data_for_pmean[0:1], (num_devices, 1))
    print(f"\n(Running pmean example on {num_devices} devices, tiling first row of data)")
else: # num_devices == data_for_pmean.shape[0]
    data_for_pmean_sharded = data_for_pmean
    print(f"\n(Running pmean example on {num_devices} devices, using all data rows)")

print(f"Input data to pmean function:\n{data_for_pmean_sharded}")

# Call the pmapped function
if num_devices > 1:
    pmapped_compute_average = pmap(compute_average_across_devices, axis_name='batch')
    local_sums, global_avg_of_sums, global_data_avg = pmapped_compute_average(data_for_pmean_sharded)
else:
    # For single device, simulate the behavior
    local_sums, global_avg_of_sums, global_data_avg = jit(compute_average_single_device)(data_for_pmean_sharded[0])
    # Wrap results to match expected shape
    local_sums = jnp.array([local_sums])
    global_avg_of_sums = jnp.array([global_avg_of_sums])
    global_data_avg = jnp.array([global_data_avg])

print(f"Local sums on each device:\n{local_sums}")
print(f"Global average of sums (same on all devices):\n{global_avg_of_sums}")
print(f"Global average of original data (element-wise, same on all devices):\n{global_data_avg}")

# Expected output for data_for_pmean if num_devices=2 (sharded: [[1,2],[3,4]]):
# Local sums: [3, 7]
# Global avg of sums: [(3+7)/2, (3+7)/2] -> [5, 5]
# Global data avg: [[(1+3)/2, (2+4)/2], ...] -> [[2, 3], [2, 3]]

# --- Part 3: A Slightly More Complex Collective Task ---
print("\n--- Part 3: A Slightly More Complex Collective Task ---")

def process_with_hybrid_output(input_shard):
    # Perform some local computation
    processed_shard = input_shard * 2.0

    # Calculate local sum
    local_sum = jnp.sum(input_shard)

    # Calculate global mean of all input data elements
    # `pmean` can average tensors, not just scalars.
    global_mean_input = pmean(input_shard, axis_name='data')

    # Return local processed shard (sharded), local sum (sharded), and global mean (replicated)
    return processed_shard, local_sum, global_mean_input

def process_with_hybrid_output_single_device(input_shard):
    # Single device version without pmean
    processed_shard = input_shard * 2.0
    local_sum = jnp.sum(input_shard)
    # For single device, global mean is just the local value
    global_mean_input = input_shard
    return processed_shard, local_sum, global_mean_input

# Prepare input data for this part
input_for_complex_task = jnp.arange(num_devices * 4, dtype=jnp.float32).reshape(num_devices, 4)
print(f"Input for complex task:\n{input_for_complex_task}")

# Call the pmapped function with specific out_axes
# (0, 0, None) means:
# - processed_shard will be sharded (leading dim 0)
# - local_sum will be sharded (leading dim 0)
# - global_mean_input will be replicated (leading dim removed)
if num_devices > 1:
    pmapped_process_hybrid = pmap(process_with_hybrid_output, axis_name='data', out_axes=(0, 0, None))
    processed_output_sharded, local_sums_sharded, global_mean_replicated = pmapped_process_hybrid(input_for_complex_task)
else:
    # For single device, simulate the behavior
    processed_output_sim, local_sums_sim, global_mean_sim = jit(process_with_hybrid_output_single_device)(input_for_complex_task[0])
    # Wrap results to match expected shape
    processed_output_sharded = jnp.array([processed_output_sim])
    local_sums_sharded = jnp.array([local_sums_sim])
    global_mean_replicated = global_mean_sim  # This should be replicated (no leading dim)

print(f"Processed output (sharded):\n{processed_output_sharded}")
print(f"Local sums (sharded):\n{local_sums_sharded}")
print(f"Global mean of input (replicated):\n{global_mean_replicated}")

print(f"Output shapes: {processed_output_sharded.shape}, {local_sums_sharded.shape}, {global_mean_replicated.shape}")

# Verify global_mean_replicated matches expected value
expected_global_mean = jnp.mean(input_for_complex_task)
print(f"Expected global mean of input (computed locally): {expected_global_mean}")

# --- Conclusion for Day 16 ---
print("\n--- Day 16 Conclusion ---")
print("Today, you've gained a deeper understanding of `jax.pmap`'s control mechanisms (`in_axes`, `out_axes`)")
print("and the power of collective operations (`axis_name`, `jax.lax.pmean`).")
print("These are essential for building robust and scalable data-parallel machine learning models in JAX.")
print("Tomorrow, we might explore other collective operations or more advanced `pmap` use cases.")