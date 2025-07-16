# Day 006: JAX's Pseudo-Random Number Generation (PRNG)

## Goal
- Understand JAX's functional approach to pseudo-random number generation using `PRNGKey`.
- Learn how to create and split keys for reproducible and independent random operations.
- See how PRNG keys interact with `jax.jit`.

## Key Learnings
- **Functional PRNG:** JAX does not use a global random state like NumPy. Instead, you explicitly manage a `PRNGKey` (a `jax.Array` representing the random state). This key is passed into functions that need to generate random numbers.
- **Reproducibility:** Starting with the same initial `PRNGKey` (seed) will always produce the same sequence of random numbers, ensuring reproducibility of your experiments.
- **Key Splitting (`jax.random.split`):** This is the most crucial concept. To generate *independent* random numbers or to pass random state to sub-functions, you *must* split your `PRNGKey`. Splitting a key generates new, derived keys that are statistically independent of each other and the original key.
    - **Avoid Key Re-use:** Passing the *same* `PRNGKey` to multiple `jax.random` functions or calling a random function multiple times with the same key will produce the *exact same* sequence of "random" numbers. This is a common pitfall.
- **`jax.random` module:** Provides a wide range of random number generation functions (`normal`, `uniform`, `randint`, `bernoulli`, etc.), all of which require a `PRNGKey` as their first argument.
- **PRNG and JIT:** When a function containing `jax.random` calls is `jit`-compiled, the key splitting and random number generation are also compiled into the XLA graph. For the JITted function to produce new random numbers on subsequent calls, you *must pass a different (newly split) `PRNGKey` into it*.

## Code Explanation (`main.py`)
- **Key Creation & Splitting:** Demonstrates how to create a master key and split it into multiple subkeys. It explicitly shows the bad practice of key re-use and the correct practice of key splitting.
- **Random Number Generation:** Examples of generating various types of random numbers (normal, uniform, integer, Bernoulli) using different `jax.random` functions.
- **PRNG with JIT:** Shows how to pass keys into JIT-compiled functions. It highlights that even within a JITted function, if you provide the same input key, you'll get the same "random" output. To get new random numbers, the input key must be new (from a split).

## Challenges/Notes
- The functional PRNG approach requires a shift in mindset from NumPy's global state. Always think about how you're passing and splitting keys.
- While `jax.random.split` is the most common, `jax.random.fold_in` is another method for incorporating new entropy, useful in specific scenarios.
- This mechanism guarantees strong reproducibility across different hardware and JAX versions when starting with the same seed.