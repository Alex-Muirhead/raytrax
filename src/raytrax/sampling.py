import jax
import jax.numpy as jnp

from raytrax.gridtypes import Array, Cell, Face, Mesh
from raytrax.intersections import LinearRay
from raytrax.random import simplex, sphere
from raytrax.transport import EXTERIOR


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


def cosine_hemisphere(key, normal: Array) -> Array:
    """Sample a cosine-weighted (Lambertian) direction on the hemisphere about `normal`.

    Exact in 3D; the 2D form is a hemisphere distribution but not the
    planar-projected cosine density.
    """
    direction = normal + sphere(key, ndim=normal.shape[-1])
    return direction / jnp.linalg.vector_norm(direction, axis=-1, keepdims=True)


def select_face_point(key, face: Face, vertices: Array):
    """Sample a point uniformly on a simplex face."""
    vertex_points = vertices[face.topology.vertices]
    num_verts = face.topology.vertices.shape[-1]
    vertex_weights = simplex(key, ndim=num_verts - 1)
    return vertex_weights @ vertex_points


def select_emission(key, cell_field: Array, face_field: Array, face_ids: Array, mesh: Mesh) -> tuple[LinearRay, Array]:
    """Sample a ray from combined volume and surface emission.

    Fields are unnormalised emitted energies (same units); `face_field`
    aligns with the boundary faces in `face_ids`. Volume emission is
    isotropic, surface emission cosine-weighted about the inward normal.
    """
    key_source, key_point, key_direction = jax.random.split(key, 3)
    num_cells = cell_field.shape[0]
    combined = jnp.concatenate([cell_field, face_field])
    source = jax.random.choice(key_source, combined.shape[0], p=combined / combined.sum())
    from_surface = source >= num_cells

    volume_cell_id = jnp.where(from_surface, 0, source)
    face_id = face_ids[jnp.where(from_surface, source - num_cells, 0)]
    face = mesh.faces[face_id]
    surface_cell_id = face.topology.cells.max()

    volume_point = select_point(key_point, mesh.cells[volume_cell_id], mesh.faces, mesh.verts)
    surface_point = select_face_point(key_point, face, mesh.verts)

    # Stored normal points outward from the parity-0 cell (column 0)
    inward_sign = jnp.where(face.topology.cells[0] == EXTERIOR, +1.0, -1.0)
    volume_tangent = sphere(key_direction, mesh.verts.shape[-1])
    surface_tangent = cosine_hemisphere(key_direction, inward_sign * face.geometry.normal)

    ray = LinearRay(
        terminus=jnp.where(from_surface, surface_point, volume_point),
        tangent=jnp.where(from_surface, surface_tangent, volume_tangent),
        travel=jnp.zeros(()),
    )
    return ray, jnp.where(from_surface, surface_cell_id, volume_cell_id)
