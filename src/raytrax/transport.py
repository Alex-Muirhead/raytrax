from enum import IntEnum

import jax
import jax.numpy as jnp
import numpy as np

from raytrax.gridtypes import Array, Mesh
from raytrax.intersections import LinearRay, crossing

# Sentinel cell index for a ray that has exited through an untagged boundary.
# Tagged boundary groups get sentinels -2, -3, ... (see grid.apply_boundaries);
# any negative cell id means the ray is terminated.
EXTERIOR = -1


class Boundary(IntEnum):
    """What a ray does when it reaches a boundary group."""

    ESCAPE = 0  # Leaves the domain, carrying its remaining energy away
    WALL = 1  # Terminates, depositing its remaining energy on the exit face
    SYMMETRY = 2  # Reflects specularly and continues in the same cell


def boundary_kinds(sentinels: dict[str, int], kinds: dict[str, Boundary]) -> Array:
    """Build the per-sentinel boundary kind table used while tracing.

    Indexed by `-sentinel - 1`, so EXTERIOR is entry 0 and the tagged groups
    follow in sentinel order. Groups left out of `kinds` escape.
    """
    table = np.full(len(sentinels) + 1, Boundary.ESCAPE, dtype=int)
    for name, kind in kinds.items():
        if name not in sentinels:
            raise ValueError(f"No boundary group named {name!r}")
        table[-sentinels[name] - 1] = kind
    return jnp.asarray(table)


def walking(cell_id: Array, ray: LinearRay, mesh: Mesh, kinds: Array):
    """Cross one cell, returning the next cell, advanced ray, distance, and exit face.

    The ray advances to the crossing point, so its parameters stay local to the
    cell. A symmetry face mirrors the tangent about its outward normal and keeps
    the ray in the same cell. A terminated ray (negative cell id) stays put and
    keeps its sentinel. The exit face id is meaningless for terminated rays.
    """
    cell = mesh.convex_cells[cell_id]
    out_face, distance = crossing(cell=cell, ray=ray)
    terminated = cell_id < 0
    # A corner or grazing hit can give a small negative distance; clamping keeps
    # the deposited energy non-negative without pushing the ray off the plane.
    distance = jnp.where(terminated, 0.0, jnp.maximum(distance, 0.0))
    neighbour_id = mesh.cells[cell_id].topology.neighbours[out_face]
    exit_face_id = mesh.cells[cell_id].topology.face_ids[out_face]

    # Sentinels index the table as -1 -> 0, -2 -> 1, ...; interior neighbours
    # clamp onto the (unused) EXTERIOR entry.
    kind = kinds[jnp.maximum(-neighbour_id - 1, 0)]
    reflects = ~terminated & (neighbour_id < 0) & (kind == Boundary.SYMMETRY)

    # convex_cells normals are unit and outward, so the mirror preserves length
    normal = cell.normal[out_face]
    reflected = ray.tangent - 2 * (ray.tangent @ normal) * normal
    next_ray = LinearRay(
        terminus=ray.terminus + distance * ray.tangent,
        tangent=jnp.where(reflects, reflected, ray.tangent),
        travel=ray.travel + distance,
    )
    next_cell_id = jnp.where(terminated | reflects, cell_id, neighbour_id)
    return next_cell_id, next_ray, distance, exit_face_id


def step(cell_id, ray: LinearRay, ray_energy, mesh: Mesh, optical_thickness, kinds: Array):
    """One continuous-forward deposition step across a single cell.

    A ray reaching a `Boundary.WALL` group deposits its remaining energy on the
    exit face; a symmetry reflection deposits nothing there.
    """
    next_cell_id, next_ray, distance, exit_face_id = walking(cell_id, ray, mesh, kinds)
    transmission = jnp.exp(-optical_thickness * distance)
    cell_deposit = ray_energy * (1 - transmission)
    remaining = ray_energy * transmission
    # cell_id >= 0 restricts to rays terminating on this step
    kind = kinds[jnp.maximum(-next_cell_id - 1, 0)]
    hit_wall = (cell_id >= 0) & (next_cell_id < 0) & (kind == Boundary.WALL)
    face_deposit = jnp.where(hit_wall, remaining, 0.0)
    remaining = jnp.where(hit_wall, 0.0, remaining)
    return next_cell_id, next_ray, remaining, cell_deposit, exit_face_id, face_deposit


def collect_step(cell_ids, rays, ray_energies, cell_energies, face_energies, mesh, optical_thickness, kinds):
    next_cell_ids, rays, ray_energies, cell_deposits, exit_face_ids, face_deposits = jax.vmap(
        step, in_axes=(0, 0, 0, None, None, None)
    )(cell_ids, rays, ray_energies, mesh, optical_thickness, kinds)
    # This *must* be *outside* of the vmap, otherwise all HELL breaks loose with allocs!
    cell_energies = cell_energies.at[cell_ids].add(cell_deposits)
    face_energies = face_energies.at[exit_face_ids].add(face_deposits)
    return next_cell_ids, rays, ray_energies, cell_energies, face_energies


def trace(cell_ids, rays: LinearRay, ray_energies, mesh: Mesh, optical_thickness, kinds: Array, num_steps: int = 100):
    """March all rays for `num_steps` cells, depositing energy continuously.

    `kinds` is the boundary kind table from `boundary_kinds`. Rays still inside
    the mesh after `num_steps` keep their remaining energy, so a non-negative
    final cell id marks a ray that ran out of steps. Returns
    (cell_ids, rays, ray_energies, cell_energies, face_energies).
    """
    num_cells = mesh.cells.geometry.volume.shape[0]
    num_faces = mesh.faces.geometry.area.shape[0]
    cell_energies = jnp.zeros(num_cells)
    face_energies = jnp.zeros(num_faces)

    def body(i, state):
        return collect_step(*state, mesh, optical_thickness, kinds)

    init_state = (cell_ids, rays, ray_energies, cell_energies, face_energies)
    return jax.lax.fori_loop(0, num_steps, body, init_state)
