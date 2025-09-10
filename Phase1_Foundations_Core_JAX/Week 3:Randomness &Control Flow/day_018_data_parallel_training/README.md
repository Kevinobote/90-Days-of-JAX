## Day 18: Building a Complete Data-Parallel Training Loop

Today marks a significant milestone: we combined all the `pmap` and collective operation concepts from the last few days into a fully functional, end-to-end data-parallel training loop. This is a fundamental pattern for scaling machine learning workloads in JAX.

### Key Concepts Applied:
1.  **State Replication**: For data-parallel training, the model parameters and optimizer state must be identical on all devices. We used `jax.tree_util.tree_map` to efficiently create a replicated copy of this state for each device.
2.  **Data Sharding**: The input data (`X_train`, `y_train`) is reshaped to have a new leading dimension equal to `num_devices`, so each device receives its own shard of the total data.
3.  **The `pmapped_train_step`**: We created a single, decorated function that handles all the per-device logic. Inside this function:
    * A local forward pass and loss calculation are performed on the data shard.
    * Local gradients are computed using `jax.grad`.
    * **`jax.lax.pmean`** is used to average the gradients across all devices. This is the crucial communication step that synchronizes the model updates.
    * The `optax` optimizer updates the replicated parameters using the averaged gradients.
4.  **Host-side Management**: The main training loop on the host simply calls the `pmapped_train_step` function, which handles all the complex parallel computation and communication. The final model parameters are then extracted from one of the replicas for a final evaluation on the host.

This pattern is the bedrock of distributed training in JAX and is highly scalable to hundreds or thousands of accelerators.