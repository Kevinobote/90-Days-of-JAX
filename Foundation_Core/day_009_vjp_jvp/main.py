# day_009_vjp_jvp/main.py

import jax
import jax.numpy as jnp

print("--- Day 9: More JAX Transformations - `jax.vjp` and `jax.jvp` ---")

# --- Part 1: Jacobian-Vector Product (JVP) with `jax.jvp` (Forward Mode AD) ---
# JVP computes the product of the Jacobian matrix and a vector.
# It's good for when you have "many inputs, few outputs" (e.g., computing sensitivity to inputs).

# Function: f(x, y) = [x^2, y^3, x*y]
# This is a vector-valued function. Its Jacobian is a matrix.
def f_vec(x, y):
    return jnp.array([x**2, y**3, x*y])

print("\n--- Jacobian-Vector Product (JVP) with `jax.jvp` ---")

# Let's consider a point (x0, y0)
x0 = jnp.array(2.0)
y0 = jnp.array(3.0)

# We want to compute JVP at (x0, y0) in the direction of (vx, vy)
vx = jnp.array(0.1)
vy = jnp.array(0.2)

# `jax.jvp(fun, primals, tangents)`
# `primals`: The point at which to evaluate the function.
# `tangents`: The vector to multiply the Jacobian by.
# Returns `(f(primals), J @ tangents)`
primal_out, tangent_out = jax.jvp(f_vec, (x0, y0), (vx, vy))

print(f"Function output f({x0}, {y0}): {primal_out}")
print(f"JVP of f at ({x0}, {y0}) in direction ({vx}, {vy}): {tangent_out}")

# Manual Calculation for verification:
# f(x, y) = [x^2, y^3, x*y]
# Jacobian J(x, y) =
# [[df1/dx, df1/dy],
#  [df2/dx, df2/dy],
#  [df3/dx, df3/dy]]
#
# J(x, y) =
# [[2x, 0  ],
#  [0,  3y^2],
#  [y,  x  ]]
#
# At (x0, y0) = (2, 3):
# J(2, 3) =
# [[2*2, 0  ],
#  [0,  3*3^2],
#  [3,  2  ]]
#
# J(2, 3) =
# [[4, 0 ],
#  [0, 27],
#  [3, 2 ]]
#
# J @ [vx, vy]^T =
# [[4, 0 ],   [0.1]   [[4*0.1 + 0*0.2 ],   [[0.4  ],
#  [0, 27], @ [0.2] =  [0*0.1 + 27*0.2], =  [5.4  ],
#  [3, 2 ]]            [3*0.1 + 2*0.2 ]]    [0.3 + 0.4]]
#
# = [0.4, 5.4, 0.7]

print(f"Manual JVP calculation: {jnp.array([0.4, 5.4, 0.7])}")
print(f"JVP result matches manual: {jnp.allclose(tangent_out, jnp.array([0.4, 5.4, 0.7]))}")


# --- Part 2: Vector-Jacobian Product (VJP) with `jax.vjp` (Reverse Mode AD) ---
# VJP computes the product of a cotangent vector (v) and the Jacobian matrix (v @ J).
# It's foundational for `jax.grad` and is efficient when you have "many outputs, few inputs"
# (e.g., computing gradients of a scalar loss function with respect to many parameters).

# Function: Same f_vec(x, y) = [x^2, y^3, x*y]
print("\n--- Vector-Jacobian Product (VJP) with `jax.vjp` ---")

# `jax.vjp(fun, *primals)`
# Returns `(f(*primals), vjp_fn)`
# `vjp_fn` is a callable that takes a cotangent vector (same shape as f's output)
# and returns the VJP (same shape as f's input).
primal_out, vjp_fn = jax.vjp(f_vec, x0, y0)

print(f"Function output f({x0}, {y0}): {primal_out}")

# Let's choose a cotangent vector (v)
v_cotangent = jnp.array([1.0, 1.0, 1.0]) # A vector with same shape as f_vec's output

# Call vjp_fn with the cotangent vector
vjp_result_x, vjp_result_y = vjp_fn(v_cotangent)

print(f"VJP of f at ({x0}, {y0}) with cotangent {v_cotangent}: ({vjp_result_x}, {vjp_result_y})")

# Manual Calculation for verification:
# We found J(2, 3) =
# [[4, 0 ],
#  [0, 27],
#  [3, 2 ]]
#
# v @ J = [1, 1, 1] @
# [[4, 0 ],
#  [0, 27],
#  [3, 2 ]]
#
# = [1*4 + 1*0 + 1*3,  1*0 + 1*27 + 1*2]
# = [4 + 0 + 3,        0 + 27 + 2]
# = [7, 29]

print(f"Manual VJP calculation: ({jnp.array(7.0)}, {jnp.array(29.0)})")
print(f"VJP result matches manual for x: {jnp.allclose(vjp_result_x, jnp.array(7.0))}")
print(f"VJP result matches manual for y: {jnp.allclose(vjp_result_y, jnp.array(29.0))}")


# --- Part 3: Relationship to `jax.grad` ---
# `jax.grad(f)(x)` is equivalent to `jax.vjp(f, x)[1](jnp.array(1.0))`

print("\n--- Relationship to `jax.grad` ---")

# Function: f(x) = x^2
def f_grad_example(x):
    return x**2

x_val = jnp.array(3.0)

# Using jax.grad
grad_val = jax.grad(f_grad_example)(x_val)
print(f"jax.grad(f)(x_val): {grad_val}")

# Using jax.vjp
# For a scalar function, the cotangent is just 1.0
_, vjp_fn_grad = jax.vjp(f_grad_example, x_val)
vjp_grad_val = vjp_fn_grad(jnp.array(1.0)) # Needs to be a JAX array 1.0

print(f"jax.vjp(f, x)[1](1.0): {vjp_grad_val[0]} (unpacking tuple)") # vjp_fn returns a tuple

print(f"grad and vjp equivalent: {jnp.allclose(grad_val, vjp_grad_val[0])}")