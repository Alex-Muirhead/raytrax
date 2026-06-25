import jax
import jax.numpy as jnp


@jax.jit(static_argnums=(1,))
def fixed_sort(arr, axis: int = -1):
    """Fixed-length sorting based on comparator sorting swaps.

    Only works for points in 1-, 2-, or 3-dimensional space.
    This means the array must have length 1--3 in the desired axis.

    Parameters
    ----------
    arr: jax.Array
        The array to sort
    axis: int = -1
        The axis to sort along. Defaults to -1

    Returns
    ------
    jax.Array
        The sorted array.
    """

    def cmp_swap(x, y):
        need_swap = x > y
        lo = jnp.where(need_swap, y, x)
        hi = jnp.where(need_swap, x, y)
        return lo, hi

    match arr.shape[axis]:
        case 1:
            sorted_arr = arr
        case 2:
            a, b = jnp.unstack(arr, axis=axis)
            a, b = cmp_swap(a, b)
            sorted_arr = jnp.stack([a, b], axis=axis)
        case 3:
            a, b, c = jnp.unstack(arr, axis=axis)
            a, b = cmp_swap(a, b)
            b, c = cmp_swap(c, b)
            a, b = cmp_swap(a, b)
            sorted_arr = jnp.stack([a, b, c], axis=axis)
        case _:
            raise ValueError("Sorting only defined for <= 3 dimensions")
    return sorted_arr


def sphere(key, ndim: int, shape: tuple[int, ...] = ()):
    """Sample a unit-sphere in N-dimensional space."""
    if not isinstance(shape, tuple):
        shape = (shape,)
    samples = jax.random.normal(key, shape=(*shape, ndim))
    samples /= jnp.linalg.vector_norm(samples, axis=-1, keepdims=True)
    return samples


def low_dimension_simplex(key, ndim: int, shape: tuple[int, ...] = ()):
    """Sample barycentric coordinates for a simplex in 1-, 2-, or 3-dimensional space."""
    samples = jax.random.uniform(key, shape=(*shape, ndim))
    sorted_samples = fixed_sort(samples)
    return jnp.diff(sorted_samples, axis=-1, prepend=0.0, append=1.0)


def high_dimension_simplex(key, ndim: int, shape: tuple[int, ...] = ()):
    """Sample barycentric coordinates for a simplex in N-dimensional space."""
    # Using notation to match CITATION
    y = jax.random.uniform(key=key, shape=(*shape, ndim))
    w = jnp.zeros((*shape, ndim + 1))
    w_sum = jnp.zeros(shape)

    # Because ndim is static, this is essentially an unrolled loop
    for d in range(ndim, 0, -1):
        y_i = y[..., d - 1]
        w_i = (1 - w_sum) * (1 - jnp.power(y_i, 1 / d))
        w = w.at[..., d - 1].set(w_i)
        w_sum += w_i

    return w.at[..., -1].set(1 - w_sum)


def simplex(key, ndim: int, shape: tuple[int, ...] = ()):
    """Sample barycentric coordinates for a simplex in N-dimensional space."""
    if not isinstance(shape, tuple):
        shape = (shape,)
    # Use a shortcut for low-dim space to save a tiny bit of time?
    if ndim <= 3:
        return low_dimension_simplex(key, ndim, shape)
    else:
        return high_dimension_simplex(key, ndim, shape)
