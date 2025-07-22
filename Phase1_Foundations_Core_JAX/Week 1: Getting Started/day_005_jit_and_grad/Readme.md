# Day 005: Combining `jax.jit` and `jax.grad`

## Goal
- Understand how to effectively combine `jax.jit` and `jax.grad` for optimized gradient computation.
- Observe the performance benefits of JIT-compiling a gradient calculation.
- See a practical example of computing gradients for a simple linear regression model.

## Key Learnings
- **Optimized Gradients:** The primary reason for combining `jax.jit` and `jax.grad` is performance. `jax.grad` generates a new function that calculates the gradient, and `jax.jit` then compiles this entire gradient computation graph into a highly optimized XLA executable.
- **Order of Application:**
    - `jax.jit(jax.grad(f))`: This means `grad(f)` is computed first (yielding a Python function for the gradient), and then *that resulting function* is JIT-compiled.
    - `jax.grad(jax.jit(f))`: This means `jit(f)` is computed first (yielding a JIT-compiled version of `f`), and then `jax.grad` differentiates this compiled function.
    - **In practice, `jax.grad(jax.jit(f))` (Option B in `main.py`) is often the more idiomatic and generally recommended approach in JAX.** When you `grad` a `jit`-compiled function, JAX's internal machinery is smart enough to trace the original function and then compute and compile its gradient graph efficiently.
- **Performance:** For numerical tasks involving gradients (like optimization in ML), `jit`-compiling the gradient computation provides significant speedups, especially for large models or data. The initial overhead includes both tracing and compilation.
- **Machine Learning Relevance:** This combination is the bedrock of training deep learning models with JAX. You define a loss function, `jit` its gradient with respect to model parameters, and then use those gradients to update parameters in an optimization loop.

## Code Explanation (`main.py`)
- **JITting Loss:** Demonstrates `jax.jit` on a basic MSE loss function, showing the familiar compilation overhead and subsequent speedup.
- **Grad of JITted Function:** Compares applying `grad` then `jit` vs. `jit` then `grad`, showing that both yield the correct result and highlighting the performance benefits of JIT on the gradient computation itself.
- **Linear Regression Example:** Provides a more complete example. It defines a `linear_model` and an `mse_loss` function. It then uses `jax.jit(jax.grad(mse_loss))` to get a highly optimized function that computes the gradients of the loss with respect to the model's parameters, given input data and true labels. This sets the stage for future optimization loops.

## Challenges/Notes
- Ensure your functions are "pure" (deterministic, no side effects) for `jax.jit` and `jax.grad` to work correctly.
- The `.block_until_ready()` method is crucial for accurate timing, as JAX operations are asynchronous.
- `jax.random.PRNGKey` is introduced briefly for data generation; we'll cover JAX's random number generation in more detail later.