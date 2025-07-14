# day_004_jax_grad/main.py

import jax
import jax.numpy as jnp

print("--- Day 4: Automatic Differentiation with `jax.grad` ---")

# --- Part 1: Gradient of a simple scalar function ---

# Define a simple function: f(x) = x^2
def f(x):
    return x**2

# Compute the gradient of f(x) with respect to x
# grad_f is now a function that computes the derivative of f
grad_f = jax.grad(f)

# Let's test it
x_val = 3.0
# Expected gradient of x^2 is 2x. So for x=3, it should be 2*3 = 6.0
gradient_at_x = grad_f(jnp.array(x_val)) # Input to grad_f must be a JAX array
print(f"\n--- Gradient of f(x) = x^2 ---")
print(f"Function f({x_val}) = {f(jnp.array(x_val))}")
print(f"Gradient of f at x={x_val}: {gradient_at_x}")
print(f"Type of gradient: {type(gradient_at_x)}")


# --- Part 2: Gradient of a slightly more complex function ---

# Define a function: g(x) = x^3 + 2x^2 + 5
# Expected gradient: g'(x) = 3x^2 + 4x
def g(x):
    return x**3 + 2*x**2 + 5

grad_g = jax.grad(g)

x_val_g = 2.0
# Expected gradient at x=2: 3*(2^2) + 4*2 = 3*4 + 8 = 12 + 8 = 20.0
gradient_at_x_g = grad_g(jnp.array(x_val_g))
print(f"\n--- Gradient of g(x) = x^3 + 2x^2 + 5 ---")
print(f"Function g({x_val_g}) = {g(jnp.array(x_val_g))}")
print(f"Gradient of g at x={x_val_g}: {gradient_at_x_g}")


# --- Part 3: Gradient with respect to multiple arguments ---
# By default, jax.grad differentiates with respect to the first argument.
# Use `argnums` to specify which arguments to differentiate with respect to.

# Define a function: h(x, y) = x^2 * y + y^3
# Partial derivative wrt x: dh/dx = 2xy
# Partial derivative wrt y: dh/dy = x^2 + 3y^2
def h(x, y):
    return x**2 * y + y**3

# Gradient wrt first argument (x)
grad_h_wrt_x = jax.grad(h, argnums=0)
# Gradient wrt second argument (y)
grad_h_wrt_y = jax.grad(h, argnums=1)
# Gradient wrt both x and y (returns a tuple of gradients)
grad_h_wrt_both = jax.grad(h, argnums=(0, 1))

x_val_h = 2.0
y_val_h = 3.0

# Test partial derivative wrt x
# Expected: 2 * 2 * 3 = 12.0
gradient_h_x = grad_h_wrt_x(jnp.array(x_val_h), jnp.array(y_val_h))
print(f"\n--- Gradient of h(x, y) = x^2 * y + y^3 ---")
print(f"h({x_val_h}, {y_val_h}) = {h(jnp.array(x_val_h), jnp.array(y_val_h))}")
print(f"Partial derivative of h wrt x at ({x_val_h}, {y_val_h}): {gradient_h_x}")

# Test partial derivative wrt y
# Expected: 2^2 + 3 * 3^2 = 4 + 3 * 9 = 4 + 27 = 31.0
gradient_h_y = grad_h_wrt_y(jnp.array(x_val_h), jnp.array(y_val_h))
print(f"Partial derivative of h wrt y at ({x_val_h}, {y_val_h}): {gradient_h_y}")

# Test gradient wrt both
gradients_h_both = grad_h_wrt_both(jnp.array(x_val_h), jnp.array(y_val_h))
print(f"Gradients of h wrt both x and y at ({x_val_h}, {y_val_h}): {gradients_h_both}")
print(f"Type of gradients_h_both: {type(gradients_h_both)}") # Should be a tuple


# --- Part 4: Requirement: Scalar Output ---
# jax.grad only works for functions that return a single scalar value.
# If your function returns an array, you typically sum it or take its mean
# for loss functions in ML.

def sum_of_squares_vec(v):
    return jnp.sum(v**2) # Returns a scalar

grad_sum_of_squares = jax.grad(sum_of_squares_vec)

vector_val = jnp.array([1.0, 2.0, 3.0])
# Expected gradient of sum(v_i^2) is [2*v_1, 2*v_2, 2*v_3]
# For [1, 2, 3], it should be [2, 4, 6]
gradient_vec = grad_sum_of_squares(vector_val)
print(f"\n--- Gradient of sum of squares of a vector ---")
print(f"Function sum_of_squares_vec({vector_val}) = {sum_of_squares_vec(vector_val)}")
print(f"Gradient at {vector_val}: {gradient_vec}")

# What if the function returned a vector?
def square_vec(v):
    return v**2 # This returns a vector [v_1^2, v_2^2, ...]

try:
    # This will raise a TypeError because the output is not scalar
    grad_square_vec = jax.grad(square_vec)
    print("\nAttempting to grad a non-scalar function (should fail):")
    grad_square_vec(jnp.array([1.0, 2.0]))
except TypeError as e:
    print(f"\nCaught expected error when trying to grad a non-scalar output: {e}")
    print("Reason: jax.grad requires the function to return a single scalar value.")