import jax
import jax.numpy as jnp
import meshio
import numpy as np
from pytest import approx

from raytrax.grid import process_cell_block
from raytrax.gridtypes import Mesh
from raytrax.intersections import LinearRay
from raytrax.transport import EXTERIOR, trace, walking


def two_quad_mesh() -> Mesh:
    """Two unit-square cells side by side, spanning (0,0) to (2,1)."""
    # fmt: off
    vertices = np.array(
        [[0.0, 0.0],
         [1.0, 0.0],
         [2.0, 0.0],
         [0.0, 1.0],
         [1.0, 1.0],
         [2.0, 1.0]],
        dtype=float
    )
    cells = np.array(
        [[0, 1, 4, 3],
         [1, 2, 5, 4]],
        dtype=int
    )
    # fmt: on
    topology = process_cell_block(meshio.CellBlock("quad", cells))
    geometry = topology.build_geometric(vertex_coords=vertices)
    return jax.tree.map(jnp.asarray, Mesh(geometry=geometry, topology=topology))


def test_convex_cells_contain_centroids():
    mesh = two_quad_mesh()
    cells = mesh.convex_cells
    centroids = mesh.geometry.cells.centroid
    assert np.all(jax.vmap(lambda cell, p: cell.contains(p))(cells, centroids))
    # A centroid is not contained in the other cell
    assert not cells[0].contains(centroids[1])


def test_walking_crosses_to_neighbour_then_exterior():
    mesh = two_quad_mesh()
    ray = LinearRay(
        terminus=jnp.array([0.5, 0.5]),
        tangent=jnp.array([1.0, 0.0]),
        travel=jnp.zeros(()),
    )
    next_cell, ray, distance = walking(jnp.array(0), ray, mesh)
    assert next_cell == 1
    assert distance == approx(0.5)
    next_cell, ray, distance = walking(next_cell, ray, mesh)
    assert next_cell == EXTERIOR
    assert distance == approx(1.0)
    assert ray.travel == approx(1.5)


def test_trace_conserves_energy():
    mesh = two_quad_mesh()
    rays = LinearRay(
        terminus=jnp.array([[0.5, 0.5], [0.5, 0.25]]),
        tangent=jnp.array([[1.0, 0.0], [0.0, -1.0]]),
        travel=jnp.zeros(2),
    )
    cell_ids = jnp.array([0, 0])
    ray_energies = jnp.array([1.0, 2.0])
    cell_ids, rays, remaining, deposited = trace(
        cell_ids, rays, ray_energies, mesh, optical_thickness=1.0, num_steps=10
    )
    assert np.all(cell_ids == EXTERIOR)
    assert remaining.sum() + deposited.sum() == approx(3.0)
