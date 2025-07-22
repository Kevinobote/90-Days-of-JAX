# day_012_control_flow/main.py

import jax
import jax.numpy as jnp
from jax import jit, lax
import time

print("--- Day 12: JIT-compatible Control Flow (`jax.lax.cond`, `jax.lax.while_loop`, `jax.lax.fori_loop`) ---")

# --- Part 1: Problem with Native Python Control Flow in JIT ---
# Python `if` and `for` are "traced" during JIT compilation.
# If their behavior depends on array *values*, it can lead to recompilations or incorrect graphs.

# @jit
# def python_if_example(x):
#     # This will work, but the *branch taken* is determined at trace time (first call).
#     # If 'x' is ever different across jitted calls, it might recompile or fail.
#     # THIS FUNCTION WILL CAUSE A TracerBoolConversionError IF x IS A JAX ARRAY
#     if x > 5.0: # <--- THIS LINE CAUSES THE ERROR WHEN x IS A TRACER
#         return x * 2
#     else:
#         return x + 10

print("\n--- Part 1: Problem with Native Python Control Flow in JIT ---")
# print(f"python_if_example(7.0): {python_if_example(jnp.array(7.0))}") # Traces for x > 5.0
# print(f"python_if_example(3.0): {python_if_example(jnp.array(3.0))}") # This line would cause the error

# More explicitly, for dynamic values, JAX traces *both* branches and uses `lax.cond` internally.
# The issue arises when control flow depends on *Python scalars derived from array values*

# @jit
# def python_loop_example(n):
#     # This loop will unroll at compile time if 'n' is a concrete (static) value.
#     # If 'n' is a JAX array whose *value* is unknown at compile time,
#     # JAX often warns about Traced<ShapedArray>, and it can lead to inefficiency
#     # or errors if the unrolling makes the graph too large.
#     result = 0
#     # The .item() here would fail if n is a tracer, or force recompilation
#     # if it manages to resolve.
#     for i in range(n.item()):
#         result += i
#     return result

# print(f"python_loop_example(5): {python_loop_example(jnp.array(5))}") # This will actually fail if n is Traced
print("Native Python control flow (if/for) depending on JAX array values inside @jit is problematic.")
print("It causes `TracerBoolConversionError` for conditionals or inefficient unrolling/errors for loops.")
print("This is why `jax.lax.cond`, `jax.lax.while_loop`, `jax.lax.fori_loop` are necessary.")


# --- Part 2: `jax.lax.cond` for Conditional Logic ---
# `lax.cond(pred, true_fn, false_fn, operand)`
# - `pred`: A boolean JAX array.
# - `true_fn`: Function to execute if pred is True.
# - `false_fn`: Function to execute if pred is False.
# - `operand`: Input(s) to pass to `true_fn` and `false_fn`.
# Important: Both `true_fn` and `false_fn` must have the same signature (take same inputs, return same outputs).

@jit
def custom_cond_example(x):
    pred = x > 5.0
    # true_fn and false_fn take 'x' as their operand
    return lax.cond(pred, lambda val: val * 2, lambda val: val + 10, x)

print("\n--- Part 2: `jax.lax.cond` for Conditional Logic ---")
print(f"custom_cond_example(7.0): {custom_cond_example(jnp.array(7.0))}")
print(f"custom_cond_example(3.0): {custom_cond_example(jnp.array(3.0))}")
print(f"custom_cond_example(5.0): {custom_cond_example(jnp.array(5.0))}") # Edge case: 5.0 > 5.0 is False, so +10


# --- Part 3: `jax.lax.while_loop` for General Loops ---
# `lax.while_loop(cond_fun, body_fun, init_val)`
# - `cond_fun`: Function `(carry) -> bool`. Determines if the loop continues.
# - `body_fun`: Function `(carry) -> carry`. The loop body, updates the carry.
# - `init_val`: Initial value of the `carry` (loop state).
# The `carry` can be a PyTree, allowing complex state.

@jit
def factorial_while(n):
    def cond_fun(carry):
        # carry = (i, acc)
        i, acc = carry
        return i <= n

    def body_fun(carry):
        # carry = (i, acc)
        i, acc = carry
        return i + 1, acc * i # Update i and acc

    init_carry = (jnp.array(1), jnp.array(1)) # (current_i, accumulator)
    final_i, final_acc = lax.while_loop(cond_fun, body_fun, init_carry)
    return final_acc

print("\n--- Part 3: `jax.lax.while_loop` for General Loops ---")
# For `while_loop`, `n` must be a scalar JAX array.
print(f"Factorial of 5: {factorial_while(jnp.array(5))}") # Expected 1*2*3*4*5 = 120
print(f"Factorial of 3: {factorial_while(jnp.array(3))}") # Expected 1*2*3 = 6


# --- Part 4: `jax.lax.fori_loop` for Fixed-Iteration Loops ---
# `lax.fori_loop(lower, upper, body_fun, init_val)`
# - `lower`: Start index (inclusive).
# - `upper`: End index (exclusive).
# - `body_fun`: Function `(i, carry) -> carry`. Loop body, takes current index `i` and `carry`.
# - `init_val`: Initial value of the `carry`.
# Ideal for fixed number of training steps/epochs inside a JIT.

@jit
def sum_range_fori(start, end):
    def body_fun(i, carry):
        return carry + i # Add current index to accumulator

    init_carry = jnp.array(0)
    # Sum from 'start' (inclusive) up to 'end' (exclusive)
    total_sum = lax.fori_loop(start, end, body_fun, init_carry)
    return total_sum

print("\n--- Part 4: `jax.lax.fori_loop` for Fixed-Iteration Loops ---")
print(f"Sum from 0 to 5 (exclusive): {sum_range_fori(0, 5)}") # Expected 0+1+2+3+4 = 10
print(f"Sum from 10 to 13 (exclusive): {sum_range_fori(10, 13)}") # Expected 10+11+12 = 33

# Example: Accumulating an array (often used in RNNs or sequential operations)
@jit
def accumulate_elements(arr):
    num_elements = arr.shape[0]
    def body_fun(i, carry):
        # carry is the accumulated sum so far
        return carry + arr[i]

    init_carry = jnp.array(0.0)
    final_sum = lax.fori_loop(0, num_elements, body_fun, init_carry)
    return final_sum

array_to_sum = jnp.array([1.0, 2.0, 3.0, 4.0])
print(f"Sum of {array_to_sum} using fori_loop: {accumulate_elements(array_to_sum)}")


# --- Part 5: Comparing Performance (Conceptual) ---
# The performance gains from using `jax.lax` primitives are most visible
# in long-running loops or complex conditionals that depend on dynamic array values.
# While Python's unrolling might be faster for *very small, fixed* loop counts,
# `jax.lax` is the way to go for general, JIT-compatible, dynamic control flow.

print("\n--- Part 5: Performance Considerations ---")
print("`jax.lax` primitives compile control flow directly into XLA.")
print("This avoids Python overhead and allows for dynamic branching/looping without recompilation.")
print("They are crucial for writing efficient and robust JAX code, especially within `jit`ted functions.")
print("For very common patterns like sequential scanning or reductions, `jax.lax.scan` is even more powerful (future topic).")