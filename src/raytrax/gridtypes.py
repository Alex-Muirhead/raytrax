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

        normal_mag = np.linalg.vector_norm(normal, axis=coord_axis)
        normal /= normal_mag[..., None]
        offset = (ref * normal).sum(axis=coord_axis)
        area = normal_mag / 2
        return FaceGeometry(area=area, normal=normal, offset=offset)


class FaceGeometry(eqx.Module):
    """Numerical values of geometric properties of each face."""

    area: Array
    normal: Array
    offset: Array


class Face(eqx.Module):
    geometry: FaceGeometry
    topology: FaceTopology


class CellTopology(eqx.Module):
    faces: Array  #      > Signed index, with sign representing face direction
    vertices: Array  #   > For random sampling
    neighbours: Array  # > Index of neighbouring cells


class CellGeometry(eqx.Module):
    """Numerical values of geometric properties of each cell."""

    volume: Array  #   > Volume of cell
    centroid: Array  # > Coordinate


class MeshTopology(eqx.Module):
    """Topological information of the mesh. Stored as indices."""

    faces: FaceTopology
    cells: CellTopology

    def build_geometric(self, vertex_coords: Array) -> MeshGeometry: ...


class MeshGeometry(eqx.Module):
    """Geometric properties of the mesh. Stored as values."""

    faces: FaceGeometry
    cells: CellGeometry
