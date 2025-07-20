# Day 003: Just-in-Time (JIT) Compilation with `jax.jit`

## Goal
- Understand the purpose and basic application of `jax.jit`.
- Observe the compilation overhead on the first execution and the performance speedup on subsequent runs.
- Begin to grasp the interaction between JIT and Python control flow, and the concept of `static_argnums`.

## Key Learnings
- **What is JIT?** `jax.jit` compiles Python functions into optimized, machine-specific code (XLA), running on your chosen device (CPU, GPU, TPU).
- **First Run Overhead:** The first time a JIT-compiled function is called with a new set of *argument shapes and types*, JAX traces the function, compiles it, and caches the compiled version. This initial run is often slower.
- **Subsequent Run Speedup:** For all subsequent calls with the *same* argument shapes and types, JAX reuses the cached compiled code, leading to significant performance improvements.
- **`block_until_ready()`:** JAX operations are asynchronous. To accurately time them, you must call `.block_until_ready()` on the result to ensure the computation has finished on the device.
- **Python Control Flow & JIT:** JIT prefers "static" control flow (conditions known at compile time). If Python `if`/`else` branches depend on *values* of arguments that change, JAX might recompile the function.
- **`static_argnums`:** For arguments whose *values* are used to control the compilation (e.g., loop counts, array shapes, boolean flags for conditional branches), you can mark them as `static_argnums` in `jax.jit`. This tells JAX to treat their values as compile-time constants, triggering recompilation *only* if that static value changes.

## Code Explanation (`main.py`)
- **Basic JIT:** Compares the execution time of a simple function with and without JIT, showcasing the initial compilation cost and subsequent speedup.
- **Control Flow:** Demonstrates how changing a boolean argument to a JIT-compiled function can trigger recompilation if JAX can't trace all paths upfront.
- **`static_argnums`:** Illustrates how `static_argnums` can be used to tell JAX that certain arguments are static (their *values* matter for compilation), leading to recompilation only when those static values change.

## Challenges/Notes
- Getting comfortable with the asynchronous nature of JAX and remembering `block_until_ready()` is important for accurate timing.
- Understanding when `jax.jit` might recompile (e.g., due to changing argument shapes/dtypes or static argument values) is crucial for performance optimization.
- This is just an introduction; we'll delve deeper into JIT's nuances, especially with JAX's `lax` control flow primitives, in later days.