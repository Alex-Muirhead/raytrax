from copy import replace

import jax
import jax.numpy as jnp

from raytrax.gridtypes import Array, Mesh
from raytrax.intersections import LinearRay, crossing

# Sentinel cell index for a ray that has exited through an untagged boundary.
# Tagged boundary groups get sentinels -2, -3, ... (see grid.apply_boundaries);
# any negative cell id means the ray is terminated.
EXTERIOR = -1


def walking(cell_id: Array, ray: LinearRay, mesh: Mesh):
    """Cross one cell, returning the next cell, advanced ray, distance, and exit face.

    A terminated ray (negative cell id) stays put and keeps its sentinel.
    The exit face id is meaningless for terminated rays.
    """
    cell = mesh.convex_cells[cell_id]
    out_face, distance = crossing(cell=cell, ray=ray)
    terminated = cell_id < 0
    distance = jnp.where(terminated, 0.0, distance)
    next_cell_id = jnp.where(terminated, cell_id, mesh.cells[cell_id].topology.neighbours[out_face])
    exit_face_id = mesh.cells[cell_id].topology.face_ids[out_face]
    next_ray = replace(ray, travel=ray.travel + distance)
    return next_cell_id, next_ray, distance, exit_face_id


def step(cell_id, ray: LinearRay, ray_energy, mesh: Mesh, optical_thickness, wall_sentinel):
    """One continuous-forward deposition step across a single cell.

    A ray crossing into `wall_sentinel` (must be negative) deposits its
    remaining energy on the exit face.
    """
    next_cell_id, next_ray, distance, exit_face_id = walking(cell_id, ray, mesh)
    transmission = jnp.exp(-optical_thickness * distance)
    cell_deposit = ray_energy * (1 - transmission)
    remaining = ray_energy * transmission
    # cell_id >= 0 restricts to rays terminating on this step
    hit_wall = (cell_id >= 0) & (next_cell_id == wall_sentinel)
    face_deposit = jnp.where(hit_wall, remaining, 0.0)
    remaining = jnp.where(hit_wall, 0.0, remaining)
    return next_cell_id, next_ray, remaining, cell_deposit, exit_face_id, face_deposit


def collect_step(cell_ids, rays, ray_energies, cell_energies, face_energies, mesh, optical_thickness, wall_sentinel):
    next_cell_ids, rays, ray_energies, cell_deposits, exit_face_ids, face_deposits = jax.vmap(
        step, in_axes=(0, 0, 0, None, None, None)
    )(cell_ids, rays, ray_energies, mesh, optical_thickness, wall_sentinel)
    # This *must* be *outside* of the vmap, otherwise all HELL breaks loose with allocs!
    cell_energies = cell_energies.at[cell_ids].add(cell_deposits)
    face_energies = face_energies.at[exit_face_ids].add(face_deposits)
    return next_cell_ids, rays, ray_energies, cell_energies, face_energies


def trace(cell_ids, rays: LinearRay, ray_energies, mesh: Mesh, optical_thickness, wall_sentinel, num_steps: int = 100):
    """March all rays for `num_steps` cells, depositing energy continuously.

    Rays hitting the (negative) `wall_sentinel` boundary deposit their
    remaining energy on the wall face. Returns
    (cell_ids, rays, ray_energies, cell_energies, face_energies).
    """
    num_cells = mesh.cells.geometry.volume.shape[0]
    num_faces = mesh.faces.geometry.area.shape[0]
    cell_energies = jnp.zeros(num_cells)
    face_energies = jnp.zeros(num_faces)

    def body(i, state):
        return collect_step(*state, mesh, optical_thickness, wall_sentinel)

    init_state = (cell_ids, rays, ray_energies, cell_energies, face_energies)
    return jax.lax.fori_loop(0, num_steps, body, init_state)
