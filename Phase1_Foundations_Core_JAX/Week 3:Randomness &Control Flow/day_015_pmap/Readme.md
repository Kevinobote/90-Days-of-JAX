# Day 015: Introduction to JAX's Parallelism with `jax.pmap`

## Goal
- Understand the concept of `jax.pmap` for automatic data parallelism across multiple devices (GPUs/TPUs).
- Learn how to use `pmap` to map a function over a leading batch dimension, distributing computation.
- Understand the role of `in_axes` in controlling how arguments are sharded or replicated.
- Implement a basic data-parallel training loop for a neural network using `pmap` and `jax.lax.pmean` for gradient aggregation.

## Key Learnings
- **`jax.pmap` (Parallel Map):** A JAX transformation that automatically maps a function over multiple devices. It's designed for **data parallelism**, where the model is replicated on each device, and different shards of data are processed in parallel.
- **`in_axes` Argument:** This is crucial for `pmap`. It controls how each input argument to the `pmap`-decorated function is handled:
    - `in_axes=0` (default for pmap inputs): The leading (0-th) axis of the input array is sharded across devices. Each device receives one slice along this axis.
    - `in_axes=None`: The entire argument is *replicated* to every device. This is typically used for model parameters, optimizer states, or other constants that are the same across all replicas.
    - `in_axes=(0, None, 0)`: For a function `f(arg1, arg2, arg3)`, `arg1` and `arg3` are sharded, `arg2` is replicated.
- **Replicas:** Each instance of the `pmap`-ed function running on a device is called a "replica". JAX manages these replicas.
- **Collective Operations (`jax.lax.pmean`, `jax.lax.all_reduce`):** For data-parallel training, after each replica computes its local gradients, these gradients need to be averaged (or summed) across all devices before parameters are updated. `jax.lax.pmean` (parallel mean) performs this efficiently. Other collectives exist for various communication patterns (e.g., `psum` for sum, `pmax` for max).
    - `pmean` requires an `axis_name` argument, which needs to be specified when `pmap` is called (e.g., `@pmap(axis_name='devices')`). The default `axis_name` is `i`.
- **Data Parallel Training Flow:**
    1.  **Initialize & Replicate:** Model parameters and optimizer state are initialized once, then replicated to all `num_devices`.
    2.  **Shard Data:** Input data (e.g., `X_batch`, `y_batch`) is split along its batch dimension, creating shards for each device.
    3.  **Pmapped Train Step:** A `pmap`-decorated function defines the per-device training logic.
        - It receives the replicated parameters/optimizer state and a shard of data.
        - Computes local loss and gradients.
        - Calls `jax.lax.pmean` on the gradients to average them across devices.
        - Updates parameters and optimizer state using the averaged gradients.
    4.  **Loop:** The training loop calls this `pmapped_train_step` for many iterations. All communication happens efficiently within JAX/XLA.
- **Output of `pmap`:** The output of a `pmap`-ed function will have a new leading axis corresponding to the devices. If a replica returns a scalar, `pmap` will return a 1D array of scalars, one for each replica.

## Code Explanation (`main.py`)
- **Part 0 (Device Check):** Reports the number of available JAX devices.
- **Part 1 (Basic `pmap`):** A simple example to show how `pmap` maps a function across a leading axis of input data.
- **Part 2 (`pmap` with `in_axes`):** Demonstrates using `in_axes=(None, 0)` to replicate parameters and shard data.
- **Part 3 (Data Parallel Training):**
    - Sets up a simple MLP, loss, and Optax.
    - **Initializes parameters and optimizer state, then explicitly `jnp.array([x] * num_devices)` them to create the leading device axis for `pmap`.** This is a common pattern to prepare replicated PyTrees for `pmap` arguments.
    - **Reshapes `X_train` and `y_train`** to have `num_devices` as their leading dimension, effectively sharding the batch.
    - The `pmapped_train_step` function is defined with `@pmap(axis_name='devices')`. Inside it, `lax.pmean(grads, axis_name='devices')` is used for gradient aggregation.
    - The main training loop calls this `pmapped_train_step`.
- **Part 4 (Gathering Results):** Shows how to extract parameters from one of the replicas after training to use them on the host or for inference.

## Challenges/Notes
- Requires multiple devices (GPUs/TPUs) to see true performance benefits. On a single device, it simulates parallelism, which can sometimes be slower than `jit` alone due to overhead.
- Data loading and sharding for `pmap` can be complex for large datasets. Libraries like `flax.linen.Partitioned` or custom data pipelines are often used.
- `pmap` works best when `num_devices` is a static value.
- This is a powerful transformation, but it introduces complexities around data sharding, parameter replication, and collective operations. It's the gateway to large-scale distributed training in JAX.