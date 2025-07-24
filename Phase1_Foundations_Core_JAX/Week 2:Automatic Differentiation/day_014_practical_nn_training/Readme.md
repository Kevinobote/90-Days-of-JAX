# Day 014: Practical Neural Network Training with Optax and JAX Primitives

## Goal
- Consolidate understanding of `jax.jit`, `jax.grad`, `jax.vmap`, PyTrees, and `optax` into a more robust and idiomatic neural network training loop.
- Demonstrate how to structure a full training process where the epoch loop is also JIT-compiled using `jax.lax.fori_loop`.
- Highlight the compositionality of JAX transformations in a practical machine learning context.

## Key Learnings
- **End-to-End JAX Training Pipeline:** This day brings together almost all the core JAX concepts covered so far into a coherent training pipeline:
    - **Data Generation:** Standard NumPy-like arrays (`jax.numpy`).
    - **Model Definition:** A simple MLP using JAX arrays and `jax.nn.relu`. Parameters are managed as PyTrees.
    - **Batched Forward Pass (`vmap`):** `jax.vmap` enables efficient parallel computation over mini-batches (or full batches in this simplified example), avoiding explicit Python loops over data.
    - **Loss Function:** Scalar loss (`jnp.mean((predictions - y_batch)**2)`), critical for `jax.grad`.
    - **Optimizer (`optax`):** A JAX-native library for optimizers (Adam, SGD, etc.) that works seamlessly with PyTrees and JAX's functional paradigm. It handles parameter updates and optimizer state management.
    - **JIT-compiled `train_step`:** The core of the training iteration, combining `jax.value_and_grad` (for loss and gradients), `optimizer.update`, and `optax.apply_updates`. This function is `jit`ted for maximum performance on accelerators.
    - **JIT-compiled Training Loop (`jax.lax.fori_loop`):** The outer loop iterating over epochs is also compiled using `jax.lax.fori_loop`. This means the *entire* training process (except for data loading/shuffling if done in Python) can reside on the accelerator, minimizing host-device communication and Python overhead.
- **`fori_loop` for Epochs:** `jax.lax.fori_loop` is well-suited for iterating a fixed number of training epochs because the number of epochs is usually known beforehand and doesn't change dynamically based on array values.
- **PyTree Management:** Parameters (`params`) and optimizer state (`opt_state`) are naturally PyTrees, allowing `jax.tree.map`, `optax.update`, and `optax.apply_updates` to operate on them seamlessly.
- **Functional State Updates:** Notice how `params` and `opt_state` are passed into `train_step` and `epoch_body_fun` as explicit arguments and returned as new values. This adheres to JAX's functional programming paradigm.
- **Performance:** By `jit`ting both the inner `train_step` and the outer `train_model` (which uses `fori_loop`), nearly the entire training process is compiled into a single XLA graph, leading to highly optimized execution.

## Code Explanation (`main.py`)
- **Parts 1-4:** Reiterate the setup for data, model, loss, and optimizer from previous days, ensuring they are JAX-idiomatic.
- **Part 5 (Training Loop):**
    - Defines `train_model` which encapsulates the multi-epoch training.
    - Uses `jax.lax.fori_loop` with `epoch_body_fun` as its core.
    - The `carry` for `fori_loop` holds `(params, opt_state, current_loss)`.
    - Calls the `jit`-compiled `train_step` inside the loop.
- **Part 6 (Visualization):** Plots the initial data and the final prediction from the trained model, showing the fit.

## Challenges/Notes
- **Logging Losses inside JIT:** Collecting *all* epoch losses directly inside a `jit`-compiled `fori_loop` or `scan` can be tricky and memory-intensive if not done carefully (e.g., using `lax.scan` to collect all outputs as an accumulated array, or only collecting sparse logs). For practical logging, you might run a small number of epochs in Python, then re-JIT for longer runs.
- **Mini-Batching:** For larger datasets, you'd typically implement mini-batching *within* each epoch. This often involves a nested `fori_loop` or `scan` that iterates through shuffled data batches. Shuffling itself also needs to be handled carefully in JAX.
- This day represents a significant milestone, showing how to build a performant and complete machine learning training pipeline in pure JAX.