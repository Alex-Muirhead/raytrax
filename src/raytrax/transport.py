from copy import replace

import jax
import jax.numpy as jnp

from raytrax.gridtypes import Array, Mesh
from raytrax.intersections import LinearRay, crossing

# Sentinel cell index for a ray that has exited the domain
EXTERIOR = -1


def walking(cell_id: Array, ray: LinearRay, mesh: Mesh):
    """Cross one cell, returning the next cell, advanced ray, and distance.

    A terminated ray (exterior cell) stays put and remains terminated.
    """
    cell = mesh.convex_cells[cell_id]
    out_face, distance = crossing(cell=cell, ray=ray)
    terminated = cell_id == EXTERIOR
    distance = jnp.where(terminated, 0.0, distance)
    next_cell_id = jnp.where(terminated, EXTERIOR, mesh.cells[cell_id].topology.neighbours[out_face])
    next_ray = replace(ray, travel=ray.travel + distance)
    return next_cell_id, next_ray, distance


def step(cell_id, ray: LinearRay, ray_energy, mesh: Mesh, optical_thickness):
    """One continuous-forward deposition step across a single cell."""
    next_cell_id, next_ray, distance = walking(cell_id, ray, mesh)
    transmission = jnp.exp(-optical_thickness * distance)
    return next_cell_id, next_ray, ray_energy * transmission, ray_energy * (1 - transmission)


def collect_step(cell_ids, rays, ray_energies, cell_energies, mesh, optical_thickness):
    new_cell_ids, rays, ray_energies, energy_dropped = jax.vmap(step, in_axes=(0, 0, 0, None, None))(
        cell_ids, rays, ray_energies, mesh, optical_thickness
    )
    # This *must* be *outside* of the vmap, otherwise all HELL breaks loose with allocs!
    cell_energies = cell_energies.at[cell_ids].add(energy_dropped)
    return new_cell_ids, rays, ray_energies, cell_energies


def trace(cell_ids, rays: LinearRay, ray_energies, mesh: Mesh, optical_thickness, num_steps: int = 100):
    """March all rays for `num_steps` cells, depositing energy continuously.

    Returns (cell_ids, rays, ray_energies, cell_energies).
    """
    num_cells = mesh.cells.geometry.volume.shape[0]
    cell_energies = jnp.zeros(num_cells)

    def body(i, state):
        return collect_step(*state, mesh, optical_thickness)

    init_state = (cell_ids, rays, ray_energies, cell_energies)
    return jax.lax.fori_loop(0, num_steps, body, init_state)
