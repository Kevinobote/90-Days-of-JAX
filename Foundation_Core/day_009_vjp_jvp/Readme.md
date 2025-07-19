# Day 009: More JAX Transformations - `jax.vjp` and `jax.jvp`

## Goal
- Understand `jax.vjp` (Vector-Jacobian Product) and `jax.jvp` (Jacobian-Vector Product) as the core primitives of JAX's automatic differentiation.
- Differentiate between forward-mode and reverse-mode AD and their typical use cases.
- See the direct relationship between `jax.grad` and `jax.vjp`.

## Key Learnings
- **Automatic Differentiation Primitives:** `jax.grad` is a high-level convenience function. Underneath, JAX uses more fundamental transformations: `jax.jvp` for forward-mode AD and `jax.vjp` for reverse-mode AD.
- **`jax.jvp` (Jacobian-Vector Product - Forward Mode AD):**
    - **Purpose:** Computes the product of the Jacobian matrix and a tangent vector. $J \cdot v$.
    - **Syntax:** `jax.jvp(fun, primals, tangents)`
        - `fun`: The function to differentiate.
        - `primals`: A tuple of the input values (the point where derivative is taken).
        - `tangents`: A tuple of vectors, one for each input, representing the direction of differentiation.
    - **Return:** A tuple `(output_primal, output_tangent)`. `output_primal` is `fun(*primals)`. `output_tangent` is $J \cdot v$.
    - **Efficiency:** Best when the number of inputs is small, and you need to compute derivatives in many directions (many `v` vectors). It's efficient for "many inputs, few outputs" scenarios.
- **`jax.vjp` (Vector-Jacobian Product - Reverse Mode AD):**
    - **Purpose:** Computes the product of a cotangent vector (adjoint/vector from the output space) and the Jacobian matrix. $v^T \cdot J$.
    - **Syntax:** `jax.vjp(fun, *primals)`
        - `fun`: The function to differentiate.
        - `*primals`: The input values (the point where derivative is taken).
    - **Return:** A tuple `(output_primal, vjp_fn)`. `output_primal` is `fun(*primals)`. `vjp_fn` is a callable that takes a cotangent vector (same shape as `fun`'s output) and returns the VJP result (same shape as `fun`'s input).
    - **Efficiency:** Best when the number of outputs is small (e.g., a single scalar loss function) and the number of inputs (parameters) is large. This is why it's the core of backpropagation in neural networks ("many outputs, few inputs" - effectively, one scalar loss for many parameters).
- **Relationship between `jax.grad` and `jax.vjp`:**
    - `jax.grad(f)(x)` is internally implemented using `jax.vjp`. Specifically, for a scalar function `f(x)`, `jax.grad(f)(x)` is equivalent to calling `jax.vjp(f, x)[1](jnp.array(1.0))`. The `jnp.array(1.0)` acts as the cotangent for a scalar output function, representing the derivative of the output with respect to itself.

## Code Explanation (`main.py`)
- **JVP Example:** Defines a vector-valued function $f(x, y)$ and demonstrates `jax.jvp` by manually verifying the result using the analytically derived Jacobian.
- **VJP Example:** Uses the same vector-valued function to show `jax.vjp`. It illustrates how `vjp_fn` takes a cotangent vector and produces the VJP, again verified by manual calculation.
- **`jax.grad` Relationship:** Explicitly shows that `jax.grad` for a scalar function is a special case of `jax.vjp` where the cotangent vector is a scalar `1.0`.

## Challenges/Notes
- `jvp` and `vjp` are more fundamental but also more complex than `grad`. `grad` is sufficient for most machine learning tasks involving scalar loss functions.
- Understanding the "shape" of the tangent and cotangent vectors is key to using `jvp` and `vjp` correctly.
- This topic dives a bit deeper into the mechanics of automatic differentiation, laying the groundwork for more advanced use cases like custom gradients, Hessian computations, or specialized AD techniques.