# Day 002: JAX vs. NumPy - Key Differences

## Goal
- Understand the fundamental concept of **immutability** in JAX arrays compared to NumPy.
- Grasp that JAX arrays are **device arrays** and operations are dispatched to accelerators (CPU, GPU, TPU).
- Observe the performance implications of JAX's JIT compilation (though full JIT understanding comes later).

## Key Learnings
- **Immutability:** Unlike NumPy, JAX arrays cannot be modified in-place. Operations that seem to modify an array (e.g., assigning a new value to an index) actually return a *new* array. Use `.at[idx].set(value)` for "updates".
- **Device Arrays:** JAX automatically manages arrays on your computational device (CPU, GPU, or TPU). When you create a `jnp.array`, it resides on the device.
- **JIT Compilation Hint:** While we don't fully dive into `jax.jit` today, the performance difference you might observe in the matrix multiplication highlights JAX's ability to compile code for faster execution, especially on subsequent runs.
- **Explicit `block_until_ready()`:** JAX operations are asynchronous. To accurately measure their execution time, you need to call `.block_until_ready()` to ensure the computation has completed on the device.

## Code Explanation (`main.py`)
- **Immutability:** Demonstrates how modifying a NumPy array changes its original ID, while an "update" on a JAX array results in a new array with a different ID, leaving the original untouched.
- **Device Arrays & Computation:** Compares the timing of a large matrix multiplication in NumPy (CPU-bound) versus JAX. It shows the initial compilation overhead and the subsequent faster execution due to JAX's device compilation.
- **Type Casting:** Briefly shows how to explicitly define the `dtype` of a JAX array, similar to NumPy.

## Challenges/Notes
- The error/warning for in-place modification on JAX arrays (`jax_array[0] = 99`) is intentional to highlight the immutability.
- You might see a significant performance boost for JAX if you have a GPU set up correctly. If not, the difference on CPU might be less dramatic but still illustrate the concept.
- Getting used to the `.at[...].set(...)` syntax for array updates is crucial in JAX.