from __future__ import annotations

import equinox as eqx
import jax
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

    def __getitem__(self, idx) -> ConvexCell:
        return ConvexCell(
            normal=self.normal[idx],
            offset=self.offset[idx],
        )


class LinearRay(eqx.Module):
    terminus: Float[Array, "... ndim"]
    tangent: Float[Array, "... ndim"]
    travel: Float[Array, "..."]

    def __check_init__(self):
        if self.terminus.shape != self.tangent.shape:
            raise ValueError("Shapes must match!")

    def __getitem__(self, idx) -> LinearRay:
        return LinearRay(
            terminus=self.terminus[idx],
            tangent=self.tangent[idx],
            travel=self.travel[idx],
        )

    @property
    def p(self) -> Float[Array, "... ndim"]:
        return self.terminus + self.travel[..., None] * self.tangent


class HyperbolicRay(eqx.Module):
    semi_major: Float[Array, "..."]
    semi_minor: Float[Array, "..."]
    linear_ecc: Float[Array, "..."]
    origin: Float[Array, "..."]

    # Only allows axis-aligned cylindrical symmetry. Reasonable enough
    @classmethod
    def from_linear(cls, ray: LinearRay, axis: int = 0) -> HyperbolicRay:
        """Construct a HyperbolicRay from a LinearRay.

        Params
        ------
        ray: `LinearRay`
            An existing ray in Cartesian space.
        axis: `int` = 0
            The axis to perform rotation around. Currently limited to one
            of 0 (x), 1 (y), and 2 (z)
        """
        # Since we are axis-aligned, just use [..., axis] to "project" down
        normal = ray.tangent.at[..., axis].set(0)
        true_ecc = 1 / jnp.linalg.vector_norm(normal, axis=-1)

        semi_major = jnp.abs(jnp.cross(ray.terminus, ray.tangent)[..., axis]) * true_ecc
        semi_minor = ray.tangent[..., axis] * semi_major * true_ecc
        linear_ecc = semi_major * true_ecc

        travel_offset = jnp.dot(ray.terminus, normal) * true_ecc**2
        origin = ray.terminus - travel_offset * ray.tangent
        # asymp_pos = jnp.stack([+semi_minor, semi_major], axis=-1)
        # asymp_neg = jnp.stack([-semi_minor, semi_major], axis=-1)

        return HyperbolicRay(
            semi_major=semi_major,
            semi_minor=semi_minor,
            linear_ecc=linear_ecc,
            origin=origin,
        )


@eqx.filter_jit
def crossing(
    cell: ConvexCell, ray: LinearRay, epsilon: float = 0.0
) -> tuple[Int[Array, "..."], Float[Array, "..."]]:
    if epsilon == 0.0:
        # We probably shouldn't use zero, due to floating precision / loss
        epsilon = np.finfo(default_floating_dtype()).resolution

    # Signed distance "outside" each half-space.
    absolute_distance = jnp.einsum("...k,...jk->...j", ray.terminus, cell.normal) - cell.offset
    alignment = jnp.einsum("...k,...jk->...j", ray.tangent, cell.normal)
    absolute_travel: Float[Array, "... nfaces"] = -absolute_distance / alignment
    # Take minimum value greater than current travel
    absolute_travel = jnp.where(alignment > epsilon, absolute_travel, jnp.nan)
    crossing_index = jnp.nanargmin(absolute_travel, axis=-1)
    # We can use jnp.nanmin, but in case we need to modify the index, we can do this!
    crossing_travel = jax.vmap(jnp.take)(absolute_travel, crossing_index) - ray.travel
    return crossing_index, crossing_travel
