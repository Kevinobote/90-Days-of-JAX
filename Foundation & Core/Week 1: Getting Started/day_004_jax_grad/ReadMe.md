# Day 004: Automatic Differentiation with `jax.grad`

## Goal
- Understand the core functionality of `jax.grad` for computing derivatives.
- Learn how to compute gradients for functions with single and multiple scalar inputs.
- Grasp the requirement that `jax.grad` only works for functions that return a single scalar value.

## Key Learnings
- **Automatic Differentiation:** `jax.grad` automatically computes the gradient (derivative for scalar functions, partial derivatives for multi-variable functions) of a given Python function.
- **Usage:** You pass a function `f` to `jax.grad(f)`, which returns a *new function* (let's call it `grad_f`). When you call `grad_f(x)`, it computes the gradient of `f` at `x`.
- **Input Requirement:** The inputs to the function being differentiated (and thus to `grad_f`) *must* be JAX arrays (`jax.Array` or `jax.numpy.ndarray`). Python floats/integers won't work.
- **`argnums`:** By default, `jax.grad` computes the gradient with respect to the *first* argument of the function. Use the `argnums` parameter (e.g., `argnums=1`, `argnums=(0, 2)`) to specify which argument(s) to differentiate with respect to. If `argnums` is a tuple, the output will also be a tuple of gradients.
- **Scalar Output Requirement:** Crucially, the function being differentiated by `jax.grad` **must return a single scalar value**. This is because `jax.grad` implements reverse-mode automatic differentiation, which is efficient for scalar-output functions (like loss functions in machine learning). If the function outputs a vector or matrix, you'll need to sum or reduce it to a scalar before applying `jax.grad` (e.g., `jnp.sum(output)`).

## Code Explanation (`main.py`)
- **Scalar Functions:** Demonstrates `jax.grad` on $f(x) = x^2$ and $g(x) = x^3 + 2x^2 + 5$, verifying the gradients against manual calculations.
- **Multi-argument Functions:** Shows how to use `argnums` to compute partial derivatives with respect to specific arguments or all arguments for $h(x, y) = x^2 * y + y^3$.
- **Scalar Output Enforcement:** Illustrates the requirement for a scalar output by showing how `jax.grad` works correctly on a sum of squares function (scalar output) but fails with a `TypeError` when applied to a function that returns a vector.

## Challenges/Notes
- Remember to always pass JAX arrays (`jnp.array()`) to the function returned by `jax.grad`.
- The scalar output requirement is a frequent point of confusion for newcomers. Always remember: `jax.grad` for loss functions, `jax.jacfwd` or `jax.jacrev` for vector-to-vector Jacobian matrices (which we will cover later).