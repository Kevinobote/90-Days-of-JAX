# day_017_collective_ops/main.py

import jax
import jax.numpy as jnp
from jax import random, pmap
from jax.lax import psum, pmean, all_gather, ppermute # Import specific collective ops
import time

print("--- Day 17: More Collective Operations with `jax.pmap` ---")

# --- Part 0: Device Information ---
print(f"\n--- Device Information ---")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")
num_devices = jax.device_count()
print(f"Number of available devices: {num_devices}")

if num_devices < 2:
    print("\nNote: `pmap` benefits from multiple devices (GPUs/TPUs).")
    print("This code will run, but parallelism benefits won't be visible on a single device.")
    print("Collective operations (`psum`, `all_gather`, `ppermute`) will act as identity or reduced operations on a single device.")

# --- Part 1: `jax.lax.psum` - Parallel Sum ---
print("\n--- Part 1: `jax.lax.psum` - Parallel Sum ---")

def compute_parallel_sum(local_value):
    # Each device has its local_value (sharded input).
    # psum sums these local_values across all devices along the 'devices' axis.
    global_sum = psum(local_value, axis_name='devices')
    return local_value, global_sum

compute_parallel_sum = pmap(compute_parallel_sum, axis_name='devices')

# Prepare sharded data
input_data_psum = jnp.arange(num_devices * 2, dtype=jnp.float32).reshape(num_devices, 2)
print(f"Input data for psum (sharded):\n{input_data_psum}")

local_values, global_sums = compute_parallel_sum(input_data_psum)

print(f"Local values on each device (input):\n{local_values}")
print(f"Global sum (same on all devices, leading dim indicates device replica):\n{global_sums}")
print(f"Shapes: local_values={local_values.shape}, global_sums={global_sums.shape}")

# Verify the result (e.g., if num_devices=2, input=[[0,1],[2,3]] -> local_sums=[[0,1],[2,3]], global_sums=[[2,4],[2,4]])
print(f"Expected global sum (computed locally for verification): {jnp.sum(input_data_psum, axis=0)}")

# --- Part 2: `jax.lax.all_gather` - Gathering Data from All Devices ---
print("\n--- Part 2: `jax.lax.all_gather` - Gathering Data from All Devices ---")

def gather_all_data(local_shard):
    # local_shard is the data piece on this device.
    # all_gather collects all local_shards from all devices along 'data_axis'
    # and concatenates them, making the full collected array available on *each* device.
    # The 'axis' argument specifies which dimension the gathered data should be added to.
    # Here, axis=0 adds it as a new leading dimension.
    gathered_data = all_gather(local_shard, axis_name='data_axis', axis=0)
    return local_shard, gathered_data

gather_all_data = pmap(gather_all_data, axis_name='data_axis')

# Prepare sharded data
input_data_gather = jnp.array([[10., 11.],
                               [20., 21.],
                               [30., 31.]], dtype=jnp.float32)

# Ensure data matches num_devices for sharding, or provide enough data
if num_devices > input_data_gather.shape[0]:
    # Tile if more devices than data rows
    data_for_gather_sharded = jnp.tile(input_data_gather[0:1], (num_devices, 1))
    print(f"(Input data for gather: Tiled first row to match {num_devices} devices)")
else:
    data_for_gather_sharded = input_data_gather[:num_devices]

print(f"Input data for all_gather (sharded):\n{data_for_gather_sharded}")

local_shards, gathered_outputs = gather_all_data(data_for_gather_sharded)

print(f"Local shard on each device (input):\n{local_shards}")
# Note: gathered_outputs's leading dimension is `num_devices` because `pmap` maps it,
# but the content of gathered_outputs[0], gathered_outputs[1], etc., should be identical.
# Let's just print the output from the first device's view for clarity.
print(f"Gathered output (full data, replicated on all devices' views):\n{gathered_outputs[0]}")
print(f"Shapes: local_shards={local_shards.shape}, gathered_outputs={gathered_outputs.shape}")

# Verify the result (if num_devices=2, input=[[10,11],[20,21]])
# gathered_outputs[0] should be [[10,11],[20,21]] if axis=0
print(f"Expected gathered output (computed locally for verification):\n{data_for_gather_sharded}")

# --- Part 3: `jax.lax.ppermute` - Peer-to-Peer Communication ---
print("\n--- Part 3: `jax.lax.ppermute` - Peer-to-Peer Communication ---")

def cyclic_shift_data(my_data):
    # This function implements a cyclic shift:
    # Each device sends its 'my_data' to the next device (sender_id)
    # and receives data from the previous device (receiver_id).

    # The device ID within the pmap is accessible via jax.lax.axis_index
    current_device_id = jax.lax.axis_index('p2p_axis')
    
    # Calculate sender and receiver IDs for a cyclic shift
    # Send to (current + 1) % num_devices
    # Receive from (current - 1 + num_devices) % num_devices
    sender_id = (current_device_id + 1) % num_devices
    receiver_id = (current_device_id - 1 + num_devices) % num_devices

    # ppermute(operand, axis_name, perm)
    # perm is a list of (source_index, target_index) pairs.
    # Here, for each device, we define where its data comes from (source)
    # and where it sends its data to (target).
    #
    # To implement a cyclic shift, device `i` sends its data to device `(i+1)%N`
    # and receives data from device `(i-1+N)%N`.
    #
    # The `perm` argument for `ppermute` is usually a list of pairs `(source_replica_id, target_replica_id)`.
    # When using `ppermute`, the function is called on *each* device.
    # Each device specifies a single `(source, target)` pair representing
    # where *its* data goes (target) and from where it expects to receive (source).
    #
    # Let's simplify and just do a "send to next" example, where each device sends its data to its neighbor.
    # The `ppermute` function returns the data received by *this* device.
    #
    # We want device `i` to receive from device `receiver_id`.
    # So, the data at `receiver_id` should be sent to `i`.
    # This means the pair is `(receiver_id, current_device_id)`.
    
    if num_devices > 1:
        shifted_data = ppermute(my_data, axis_name='p2p_axis', perm=[(receiver_id, current_device_id)])
    else:
        # On a single device, ppermute effectively returns its input.
        shifted_data = my_data
    
    return my_data, shifted_data, current_device_id

cyclic_shift_data = pmap(cyclic_shift_data, axis_name='p2p_axis')

# Prepare sharded data. Use simple integers for clarity.
input_data_p2p = jnp.arange(num_devices, dtype=jnp.float32).reshape(num_devices, 1) # (num_devices, 1)
print(f"Input data for ppermute (sharded):\n{input_data_p2p}")

local_original_data, received_data, device_ids = cyclic_shift_data(input_data_p2p)

print(f"Original data on each device:\n{local_original_data}")
print(f"Data received by each device (after cyclic shift):\n{received_data}")
print(f"Device IDs:\n{device_ids}")

# Verification for num_devices = 3, input = [[0],[1],[2]]
# Device 0 (id 0) receives from device 2. Expected received: [2]
# Device 1 (id 1) receives from device 0. Expected received: [0]
# Device 2 (id 2) receives from device 1. Expected received: [1]
# This should result in received_data = [[2],[0],[1]]
if num_devices > 1:
    expected_received_data = jnp.roll(input_data_p2p, shift=1, axis=0) # Shift by 1 along the device axis
    print(f"Expected received data (local roll for verification):\n{expected_received_data}")
else:
    print("(ppermute on single device returns original data)")

# --- Conclusion for Day 17 ---
print("\n--- Day 17 Conclusion ---")
print("You've now explored three more vital collective operations in JAX:")
print("- `psum`: For summing values across devices.")
print("- `all_gather`: For replicating collected sharded data onto all devices.")
print("- `ppermute`: For fine-grained peer-to-peer data exchange.")
print("These primitives allow for complex communication patterns necessary for advanced distributed algorithms.")
print("Tomorrow, we might explore practical applications of these, or delve into JAX's distributed setup.")