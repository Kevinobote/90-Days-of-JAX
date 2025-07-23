# day_013_lax_scan/main.py

import jax
import jax.numpy as jnp
from jax import jit, lax

print("--- Day 13: 'jax.lax.scan' for Efficient Recurrence and Accumulation ---")

# --- Part 1: Understanding `jax.lax.scan` ---
# `jax.lax.scan` is a powerful primitive for efficiently expressing loops over sequences.
# It's like a functional "fold" or "reduce" operation.
# `lax.scan(f, init, xs, length=None)`
# - `f`: The scan function (callable) with signature `(carry, x) -> (carry, y)`.
#        `carry`: The accumulated state passed from one iteration to the next.
#        `x`: An element from the input sequence `xs`.
#        `y`: An element of the output sequence.
# - `init`: The initial value of the `carry`.
# - `xs`: The sequence (array or PyTree of arrays) over which to scan. Its leading dimension
#         is iterated over. If `None`, `f` is called `length` times with `x=None`.
# - `length`: (Optional) The number of iterations. Inferred from `xs` if provided.
#
# Returns `(final_carry, ys)`
# - `final_carry`: The value of `carry` after the last iteration.
# - `ys`: A PyTree of arrays containing all `y` outputs stacked along a new leading dimension.

print("\n--- Part 1: Understanding `jax.lax.scan` ---")

# Example 1: Simple accumulation (summing elements of an array)
# Equivalent to sum_range_fori from Day 12, but more idiomatic for JAX
@jit
def sum_with_scan(arr):
    def body_fun(carry, x):
        # carry: current sum
        # x: current element from arr
        new_carry = carry + x
        output = None # We don't need individual outputs, just the final sum
        return new_carry, output # (new_carry, output_for_iteration)
    
    initial_sum = jnp.array(0.0)
    # xs=arr will cause 'x' in body_fun to be each element of arr
    final_sum, _ = lax.scan(body_fun, initial_sum, arr)
    return final_sum

array_to_sum = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Sum of {array_to_sum}: using scan: {sum_with_scan(array_to_sum)}")
print(f"Expected sum: {jnp.sum(array_to_sum)}") # Should be 15.0

# Example 2: Computing a running product and collecting intermediate results
@jit
def running_product_with_scan(arr):
    def body_fun(carry, x):
        # carry: (running_product, intermediate_results)
        # x: current element from arr
        new_carry = carry * x
        output = new_carry # We want to collect each intermediate product
        return new_carry, output
    
    initial_product = jnp.array(1.0)
    final_product, intermediate_products = lax.scan(body_fun, initial_product, arr)
    return final_product, intermediate_products

prod_array = jnp.array([2.0, 3.0, 4.0])
final_prod, intermediate_prods = running_product_with_scan(prod_array)
print(f"\nRunning product of {prod_array}:")
print(f"Final product: {final_prod}") # Expected 24.0
print(f"Intermediate products: {intermediate_prods}") # Expected [1, 2, 6, 24]

# --- Part 2: `scan` for Recurrent Neural Networks (RNNs) - Conceptual Example ---
# `scan` is the canonical way to implement the core recurrence in RNNs, LSTMs, GRUs in JAX.
# (Simplified example, not a full RNN model)

@jit
def simple_rnn_scan(params, initial_state, inputs):
    # params: PyTree of RNN weights/biases
    # initial_state: initial hidden state (e.g., jnp.zeros(hidden_dim))
    # inputs: sequence of input vectors (num_timesteps, input_dim)

    # The body function for one time step
    def rnn_step(carry_state, input_t):
        # carry_state: hidden state from previous time step
        # input_t: current input at time t
        # For simplicity, let's say h_t = relu(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
        h_prev = carry_state
        # Assuming params has 'W_hh', 'W_xh', 'b_h' for a simple linear transformation
        # (In a real RNN, you'd define this more properly)
        new_h = jnp.dot(h_prev, params['W_hh']) + jnp.dot(input_t, params['W_xh']) + params['b_h']
        new_h = jax.nn.relu(new_h) # Activation function

        # output_t can be anything you want to collect (e.g., new_h, or a separate output layer)
        output_t = new_h # Let's say we output the hidden state at each step

        return new_h, output_t #(new_carry_state, output_for_this_timestep)
    
    # The scan operates over the 'inputs' sequence
    final_state, all_outputs = lax.scan(rnn_step, initial_state, inputs)
    return final_state, all_outputs

print("\n--- Part 2: `scan` for Recurrent Neural Networks (RNNs) - Conceptual Example ---")

# Define dummy parameters and data for the conceptual RNN
key = jax.random.PRNGKey(0)
input_dim = 10
hiden_dim = 20
num_timesteps = 5

key, w_hh_key, w_xh_key, b_h_key = jax.random.split(key, 4)
rnn_params = {
    'W_hh': jax.random.normal(w_hh_key, (hiden_dim, hiden_dim)),
    'W_xh': jax.random.normal(w_xh_key, (input_dim, hiden_dim)),
    'b_h': jax.random.normal(b_h_key, (hiden_dim,))
}

initial_rnn_state = jnp.zeros(hiden_dim)
# Batch of inputs (num_timesteps, input_dim)
key, input_key = jax.random.split(key)
rnn_inputs = jax.random.normal(input_key, (num_timesteps, input_dim))

final_rnn_state, rnn_outputs = simple_rnn_scan(rnn_params, initial_rnn_state, rnn_inputs)

print(f"Initial RNN state shape: {initial_rnn_state.shape}")
print(f"RNN Inputs shape: {rnn_inputs.shape}")
print(f"Final RNN state shape: {final_rnn_state.shape}")
print(f"All RNN outputs shape (num_timesteps, hidden_dim): {rnn_outputs.shape}")
print(f"First output of RNN outputs:\n{rnn_outputs[0][:5]}...") # Print first 5 elements of first output
print("This demonstrates how `scan` can efficiently process sequences, making it ideal for RNNs.")


# --- Part 3: `scan` vs. `fori_loop` vs. Python Loop (Conceptual Performance) ---
# - Python loop: Slow due to interpreter overhead, not JIT-compatible for dynamic lengths.
# - `fori_loop`: Good for simple fixed iterations, but manually manage accumulation.
# - `scan`: Optimized for *recurrence* and *accumulation of intermediate results*.
#          JAX can apply optimizations specific to recurrent patterns.

print("\n--- Part 3: `scan` vs. `fori_loop` vs. Python Loop (Conceptual) ---")
print("`jax.lax.scan` is often preferred over `fori_loop` for recurrent computations")
print("or when accumulating results across iterations, as it's designed for these patterns.")
print("It can also be more memory efficient than manual accumulation with `fori_loop` for large `ys`.")
print("JAX's compiler can apply specific optimizations to `scan` that it cannot to general `fori_loop` or `while_loop` structures.")

