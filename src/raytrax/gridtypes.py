from typing import Self

import equinox as eqx
import jax
import numpy as np

Array = jax.Array | np.typing.NDArray


class IndexingMixin:
    def __getitem__(self, idx) -> Self:
        return jax.tree.map(lambda leaf: leaf[idx], self)


class FaceTopology(eqx.Module, IndexingMixin):
    """Topological (index) information about face/s."""

    vertices: Array
    cells: Array

    def build_geometric(self, vertex_coords: Array) -> FaceGeometry:
        vertex_axis, coord_axis = -2, -1
        face_coords = vertex_coords[self.vertices, :]

        match face_coords.shape[vertex_axis]:
            case 2:
                ref, u = np.unstack(face_coords, axis=vertex_axis)
                u = u - ref
                # equal to -Iu, where I is pseudoscalar
                normal = u @ np.array([[0, -1], [+1, 0]])
            case 3:
                ref, u, v = np.unstack(face_coords, axis=vertex_axis)
                u = u - ref
                v = v - ref
                # equal to -I(u^v) / 2, where I is pseudoscalar
                normal = np.cross(u, v) / 2
            case _:
                raise NotImplementedError("Only 1- and 2-simplex faces are supported")

        centroid = face_coords.mean(axis=vertex_axis)  # Only true for simplices
        area = np.linalg.vector_norm(normal, axis=coord_axis)
        normal /= area[..., None]
        offset = (ref * normal).sum(axis=coord_axis)
        return FaceGeometry(centroid=centroid, area=area, normal=normal, offset=offset)


class FaceGeometry(eqx.Module, IndexingMixin):
    """Numerical values of geometric properties of each face."""

    centroid: Array
    area: Array
    normal: Array
    offset: Array


class Face(eqx.Module, IndexingMixin):
    geometry: FaceGeometry
    topology: FaceTopology


class CellTopology(eqx.Module, IndexingMixin):
    face_ids: Array  # (n_cells, n_faces_per_cell) face indices into FaceTopology
    face_signs: Array  # (n_cells, n_faces_per_cell) +/-1; +1 = canonical normal is outward, -1 = inward
    vertices: Array  # > For random sampling
    neighbours: Array  # > Index of neighbouring cells (-1 = boundary)

    def build_geometric(self, face_geom: FaceGeometry) -> CellGeometry:
        vector_vertex_axis, vector_coord_axis = -2, -1
        scalar_vertex_axis = -1
        ndim = face_geom.centroid.shape[vector_coord_axis]

        # Until we have nice indexing on our PyTrees
        faces: FaceGeometry = jax.tree.map(lambda leaf: leaf[self.face_ids], face_geom)

        # Volume and centroid via divergence theorem.
        # For a convex cell: d*V = sum_f (outward_offset_f * A_f^true)
        # => V = (1/d) * sum(outward_offset * area_stored)
        # => C = (1/(d+1)) * sum(outward_offset * face_centroid * area_stored) / V

        volume_contributions = self.face_signs * faces.offset * faces.area
        centroid_contributions = faces.centroid * volume_contributions[..., None]

        # NOTE: These are *not* averages! They are based on divergence theorem
        volume = (1 / ndim) * volume_contributions.sum(axis=scalar_vertex_axis)
        centroid = (1 / (ndim + 1)) * centroid_contributions.sum(axis=vector_vertex_axis) / volume[..., None]

        # Distance from face to centroid?
        # Broadcast (num_cells, num_dims) -> (num_cells, num_faces, num_dims)
        local_offset = ((faces.centroid - centroid[..., None, :]) * faces.normal).sum(axis=vector_coord_axis)
        face_weights = (1 / ndim) * self.face_signs * local_offset * faces.area
        # Normalise
        face_weights /= volume[..., None]

        return CellGeometry(volume=volume, centroid=centroid, face_weights=face_weights)


class CellGeometry(eqx.Module, IndexingMixin):
    """Numerical values of geometric properties of each cell."""

    volume: Array  #   > Volume of cell
    centroid: Array  # > Coordinate
    face_weights: Array  # (n_cells, n_faces_per_cell)


class Cell(eqx.Module, IndexingMixin):
    geometry: CellGeometry
    topology: CellTopology


class MeshTopology(eqx.Module, IndexingMixin):
    """Topological information of the mesh. Stored as indices."""

    faces: FaceTopology
    cells: CellTopology

    def build_geometric(self, vertex_coords: Array) -> MeshGeometry:
        face_geom = self.faces.build_geometric(vertex_coords)
        cell_geom = self.cells.build_geometric(face_geom)
        return MeshGeometry(faces=face_geom, cells=cell_geom, verts=vertex_coords)


class MeshGeometry(eqx.Module, IndexingMixin):
    """Geometric properties of the mesh. Stored as values."""

    faces: FaceGeometry
    cells: CellGeometry
    verts: Array


class Mesh(eqx.Module, IndexingMixin):
    geometry: MeshGeometry
    topology: MeshTopology

    @property
    def faces(self) -> Face:
        return Face(
            geometry=self.geometry.faces,
            topology=self.topology.faces,
        )

    @property
    def cells(self) -> Face:
        return Cell(
            geometry=self.geometry.cells,
            topology=self.topology.cells,
        )

    @property
    def verts(self) -> Array:
        return self.geometry.verts
