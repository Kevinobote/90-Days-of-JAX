# Day 001: JAX Installation and Basic Array Operations

## Goal
- Successfully install JAX.
- Create and inspect JAX arrays (`jax.Array`).
- Perform fundamental arithmetic operations with JAX arrays.

## Key Learnings
- JAX arrays are immutable, similar to NumPy arrays but with a different underlying implementation for JIT compilation.
- `jax.numpy` (aliased as `jnp`) provides a NumPy-like API for JAX arrays.
- JAX automatically leverages your available hardware (CPU, GPU, TPU) without explicit configuration (though GPU/TPU setup requires specific installation).

## Code Explanation (`main.py`)
The `main.py` script demonstrates:
1. Verification of JAX version and default backend.
2. Creation of a `jnp.array` and printing its type, shape, and dtype.
3. Examples of basic arithmetic: addition, multiplication, division (scalar and element-wise), dot product, and sum of squares.

## Challenges/Notes
- Ensure your Python environment is set up correctly (e.g., using a virtual environment).
- If you encounter issues with GPU/TPU installation, start with the CPU-only version first to get the basics working.
- Notice that operations on JAX arrays return *new* arrays, reflecting their immutability.