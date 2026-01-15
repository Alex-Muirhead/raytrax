import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from equinox._misc import default_floating_dtype


class ConvexCell(eqx.Module):
    normal: Float[Array, "... nfaces ndim"]
    offset: Float[Array, "... nfaces"]

    @eqx.filter_jit
    def contains(self, points: Float[Array, "... ndim"], epsilon: float = 0.0):
        if epsilon == 0.0:
            epsilon = np.finfo(default_floating_dtype()).resolution
        # Signed distance "outside" each half-space.
        distance = jnp.einsum("...k,...jk->...j", points, self.normal) - self.offset
        return jnp.all(distance <= epsilon, axis=-1)


class LinearRay(eqx.Module):
    terminus: Float[Array, "... ndim"]
    tangent: Float[Array, "... ndim"]
    travel: Float[Array, "..."]

    def __check_init__(self):
        if self.terminus.shape != self.tangent.shape:
            raise ValueError("Shapes must match!")


# @eqx.filter_jit
def crossing(
    cell: ConvexCell, ray: LinearRay, epsilon: float = 0.0
) -> tuple[Int[Array, "..."], Float[Array, "..."]]:
    if epsilon == 0.0:
        # We probably shouldn't use zero, due to floating precision / loss
        epsilon = np.finfo(default_floating_dtype()).resolution

    # Signed distance "outside" each half-space.
    absolute_distance = jnp.einsum("...k,...jk->...j", ray.terminus, cell.normal) - cell.offset
    alignment = jnp.einsum("...k,...jk->...j", ray.tangent, cell.normal)
    relative_distance = absolute_distance + alignment * ray.travel
    absolute_travel: Float[Array, "... nfaces"] = -absolute_distance / alignment
    # Take minimum value greater than current travel
    absolute_travel = jnp.where(alignment > epsilon, absolute_travel, jnp.nan)
    print(relative_distance, relative_distance > epsilon)
    crossing_index: Int[Array, "..."] = jnp.where(
        jnp.any(relative_distance > epsilon, axis=-1),
        jnp.nanargmax(relative_distance, axis=-1),
        jnp.nanargmin(absolute_travel, axis=-1),
    )
    crossing_travel: Float[Array, "..."] = jnp.maximum(
        absolute_travel[..., crossing_index] - ray.travel, 0
    )
    return crossing_index, crossing_travel
