import jax.numpy as jnp
import jax


def sort3_with_parity_bit(arr):
    """Sorting for 3 elements, returns (sorted, parity)."""
    a, b, c = arr
    swaps = 0

    def cmp_swap(x, y, s):
        need_swap = x > y
        lo = jnp.where(need_swap, y, x)
        hi = jnp.where(need_swap, x, y)
        return lo, hi, s + jnp.where(need_swap, 1, 0)

    a, b, swaps = cmp_swap(a, b, swaps)
    b, c, swaps = cmp_swap(b, c, swaps)
    a, b, swaps = cmp_swap(a, b, swaps)

    sorted_arr = jnp.stack([a, b, c])
    parity = swaps % 2
    return sorted_arr, parity
