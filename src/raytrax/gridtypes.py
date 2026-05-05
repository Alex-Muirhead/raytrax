import equinox as eqx
import jax
import numpy as np

Array = jax.Array | np.typing.NDArray


class FaceTopology(eqx.Module):
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
                normal = u @ np.array([[0, -1], [+1, 0]])
            case 3:
                ref, u, v = np.unstack(face_coords, axis=vertex_axis)
                u = u - ref
                v = v - ref
                normal = np.cross(u, v)
            case _:
                raise NotImplementedError("Only 1- and 2-simplex faces are supported")

        centroid = face_coords.mean(axis=vertex_axis)
        normal_mag = np.linalg.vector_norm(normal, axis=coord_axis)
        normal /= normal_mag[..., None]
        offset = (ref * normal).sum(axis=coord_axis)
        area = normal_mag / 2
        return FaceGeometry(centroid=centroid, area=area, normal=normal, offset=offset)


class FaceGeometry(eqx.Module):
    """Numerical values of geometric properties of each face."""

    centroid: Array
    area: Array
    normal: Array
    offset: Array


class Face(eqx.Module):
    geometry: FaceGeometry
    topology: FaceTopology


class CellTopology(eqx.Module):
    face_idx: Array   # (n_cells, n_faces_per_cell) face indices into FaceTopology
    face_sgn: Array   # (n_cells, n_faces_per_cell) +/-1; -1 = canonical normal is outward, +1 = inward
    vertices: Array   # > For random sampling
    neighbours: Array  # > Index of neighbouring cells (-1 = boundary)

    def build_geometric(self, face_geom: FaceGeometry) -> CellGeometry:
        ndim = face_geom.centroid.shape[-1]

        # Volume and centroid via divergence theorem.
        # For a convex cell: d*V = sum_f (outward_offset_f * A_f^true)
        # outward_offset = -face_sgn * canonical_offset  (face_sgn=-1 means outward)
        # A_f^true = 2/(d-1) * area_stored  (corrects for the /2 in build_geometric)
        # => V = (2/(d*(d-1))) * sum(outward_offset * area_stored)
        # => C = (2/((d-1)*(d+1))) * sum(outward_offset * face_centroid * area_stored) / V
        canonical_offsets = face_geom.offset[self.face_idx]     # (n_cells, n_faces)
        face_areas = face_geom.area[self.face_idx]               # (n_cells, n_faces)
        face_centroids = face_geom.centroid[self.face_idx]       # (n_cells, n_faces, ndim)
        outward_offsets = -self.face_sgn * canonical_offsets     # (n_cells, n_faces)

        volume = (2 / (ndim * (ndim - 1))) * (outward_offsets * face_areas).sum(axis=-1)

        centroid_num = (outward_offsets[..., None] * face_centroids * face_areas[..., None]).sum(axis=-2)
        centroid = (2 / ((ndim - 1) * (ndim + 1))) * centroid_num / volume[..., None]

        return CellGeometry(volume=volume, centroid=centroid)


class CellGeometry(eqx.Module):
    """Numerical values of geometric properties of each cell."""

    volume: Array  #   > Volume of cell
    centroid: Array  # > Coordinate


class MeshTopology(eqx.Module):
    """Topological information of the mesh. Stored as indices."""

    faces: FaceTopology
    cells: CellTopology

    def build_geometric(self, vertex_coords: Array) -> MeshGeometry:
        face_geom = self.faces.build_geometric(vertex_coords)
        cell_geom = self.cells.build_geometric(face_geom)
        return MeshGeometry(faces=face_geom, cells=cell_geom)


class MeshGeometry(eqx.Module):
    """Geometric properties of the mesh. Stored as values."""

    faces: FaceGeometry
    cells: CellGeometry
