import jax
import jax.numpy as jnp

from raytrax.gridtypes import Array, Cell, Face, Mesh
from raytrax.intersections import LinearRay
from raytrax.random import simplex, sphere


def select_point(key, cell: Cell, faces: Face, vertices: Array):
    """Sample a point uniformly within a convex cell.

    A sub-simplex (cell centroid + face vertices) is chosen with probability
    proportional to its volume fraction, then a barycentric point within it.
    """
    subcell_key, barycentric_key = jax.random.split(key, num=2)
    face_id = jax.random.choice(key=subcell_key, a=cell.topology.face_ids, p=cell.geometry.face_weights)
    vertex_points = vertices[faces[face_id].topology.vertices]
    ndim = vertices.shape[-1]
    vertex_weights = simplex(key=barycentric_key, ndim=ndim)
    return vertex_weights[0] * cell.geometry.centroid + vertex_weights[1:] @ vertex_points


def select_start(key, field: Array, mesh: Mesh) -> tuple[LinearRay, Array]:
    """Sample a ray origin with probability proportional to `field`, and a
    direction uniform on the sphere."""
    key_cell, key_point, key_direction = jax.random.split(key, 3)
    num_cells = field.shape[0]
    ndim = mesh.verts.shape[-1]
    cell_id = jax.random.choice(key_cell, num_cells, p=field)
    terminus = select_point(key_point, mesh.cells[cell_id], mesh.faces, mesh.verts)
    tangent = sphere(key_direction, ndim)
    ray = LinearRay(terminus=terminus, tangent=tangent, travel=jnp.zeros(()))
    return ray, cell_id
