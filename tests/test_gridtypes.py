import numpy as np
import pytest
from pytest import approx

from raytrax.gridtypes import CellTopology, FaceTopology


def make_topo(vertex_indices):
    return FaceTopology(
        vertices=np.array(vertex_indices),
        cells=np.zeros(0, dtype=int),
    )


def make_face_geom(vertex_coords, face_vertex_indices):
    return make_topo(face_vertex_indices).build_geometric(vertex_coords)


def make_cell_topo(face_idx, face_sgn):
    face_idx = np.asarray(face_idx)
    return CellTopology(
        face_idx=face_idx,
        face_sgn=np.asarray(face_sgn),
        vertices=np.zeros(0, dtype=int),
        neighbours=np.full_like(face_idx, -1),
    )


class TestFaceGeometry2D:
    def test_horizontal_edge_normal(self):
        # u=(1,0) → rotated 90° clockwise → (0,-1)
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        geom = make_topo([0, 1]).build_geometric(coords)
        assert np.allclose(geom.normal, [0.0, -1.0])

    def test_vertical_edge_normal(self):
        # u=(0,1) → rotated 90° clockwise → (1,0)
        coords = np.array([[0.0, 0.0], [0.0, 1.0]])
        geom = make_topo([0, 1]).build_geometric(coords)
        assert np.allclose(geom.normal, [1.0, 0.0])

    def test_normal_is_unit_vector(self):
        coords = np.array([[1.0, 2.0], [4.0, 6.0]])
        geom = make_topo([0, 1]).build_geometric(coords)
        assert float(np.linalg.norm(geom.normal)) == approx(1.0)

    def test_unit_edge_area(self):
        # area = edge_length / 2
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        geom = make_topo([0, 1]).build_geometric(coords)
        assert float(geom.area) == approx(0.5)

    def test_longer_edge_area(self):
        # Length-4 edge: area = 4/2 = 2
        coords = np.array([[0.0, 0.0], [4.0, 0.0]])
        geom = make_topo([0, 1]).build_geometric(coords)
        assert float(geom.area) == approx(2.0)

    def test_offset_at_origin(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        geom = make_topo([0, 1]).build_geometric(coords)
        assert float(geom.offset) == approx(0.0)

    def test_offset_not_at_origin(self):
        # Edge (1,1)→(2,1): normal=(0,-1), offset=(1,1)·(0,-1)=-1
        coords = np.array([[1.0, 1.0], [2.0, 1.0]])
        geom = make_topo([0, 1]).build_geometric(coords)
        assert float(geom.offset) == approx(-1.0)

    def test_offset_satisfies_plane_equation(self):
        # Both endpoints must lie on the plane: normal·p = offset
        coords = np.array([[3.0, 1.0], [3.0, 4.0]])
        geom = make_topo([0, 1]).build_geometric(coords)
        for pt in coords:
            assert float(geom.normal @ pt) == approx(float(geom.offset))

    def test_batched_square_edges(self):
        # Three edges of a unit square
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        geom = make_topo([[0, 1], [1, 2], [2, 3]]).build_geometric(coords)
        # (0,0)→(1,0): normal=(0,-1), offset=0
        assert np.allclose(geom.normal[0], [0.0, -1.0])
        assert float(geom.offset[0]) == approx(0.0)
        # (1,0)→(1,1): normal=(1,0), offset=1
        assert np.allclose(geom.normal[1], [1.0, 0.0])
        assert float(geom.offset[1]) == approx(1.0)
        # (1,1)→(0,1): normal=(0,1), offset=1
        assert np.allclose(geom.normal[2], [0.0, 1.0])
        assert float(geom.offset[2]) == approx(1.0)
        assert np.allclose(geom.area, [0.5, 0.5, 0.5])


class TestFaceGeometry3D:
    def test_xy_plane_triangle_normal(self):
        # cross((1,0,0), (0,1,0)) = (0,0,1)
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        geom = make_topo([0, 1, 2]).build_geometric(coords)
        assert np.allclose(geom.normal, [0.0, 0.0, 1.0])

    def test_yz_plane_triangle_normal(self):
        # cross((0,1,0), (0,0,1)) = (1,0,0)
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        geom = make_topo([0, 1, 2]).build_geometric(coords)
        assert np.allclose(geom.normal, [1.0, 0.0, 0.0])

    def test_normal_is_unit_vector(self):
        coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
        geom = make_topo([0, 1, 2]).build_geometric(coords)
        assert float(np.linalg.norm(geom.normal)) == approx(1.0)

    def test_unit_right_triangle_area(self):
        # Legs of length 1: area = 0.5
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        geom = make_topo([0, 1, 2]).build_geometric(coords)
        assert float(geom.area) == approx(0.5)

    def test_larger_triangle_area(self):
        # Legs of length 2: area = |cross((2,0,0),(0,2,0))| / 2 = 4/2 = 2
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        geom = make_topo([0, 1, 2]).build_geometric(coords)
        assert float(geom.area) == approx(2.0)

    def test_offset_at_origin(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        geom = make_topo([0, 1, 2]).build_geometric(coords)
        assert float(geom.offset) == approx(0.0)

    def test_offset_not_at_origin(self):
        # Triangle in the plane x=1: normal=(1,0,0), offset=1
        coords = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        geom = make_topo([0, 1, 2]).build_geometric(coords)
        assert float(geom.offset) == approx(1.0)

    def test_offset_satisfies_plane_equation(self):
        # All three vertices must lie on the plane: normal·p = offset
        coords = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        geom = make_topo([0, 1, 2]).build_geometric(coords)
        for pt in coords:
            assert float(geom.normal @ pt) == approx(float(geom.offset))

    def test_batched_parallel_faces(self):
        # Bottom (z=0) and top (z=1) right triangles of a unit cube
        coords = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
        ])
        geom = make_topo([[0, 1, 2], [3, 4, 5]]).build_geometric(coords)
        assert np.allclose(geom.normal[0], [0.0, 0.0, 1.0])
        assert float(geom.offset[0]) == approx(0.0)
        assert np.allclose(geom.normal[1], [0.0, 0.0, 1.0])
        assert float(geom.offset[1]) == approx(1.0)
        assert np.allclose(geom.area, [0.5, 0.5])


class TestFaceGeometryErrors:
    def test_raises_for_quad_face(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        with pytest.raises(NotImplementedError):
            make_topo([0, 1, 2, 3]).build_geometric(coords)


class TestCellGeometry2D:
    def test_unit_square(self):
        # CCW quad with vertices (0,0),(1,0),(1,1),(0,1)
        # Edge windings: (0,1),(1,2),(2,3),(3,0); sort parities [0,0,0,1] -> sgn [-1,-1,-1,+1]
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        face_geom = make_face_geom(coords, [[0, 1], [1, 2], [2, 3], [0, 3]])
        cell_topo = make_cell_topo(
            face_idx=[[0, 1, 2, 3]],
            face_sgn=[[-1, -1, -1, +1]],
        )
        cell_geom = cell_topo.build_geometric(face_geom)
        assert float(cell_geom.volume[0]) == approx(1.0)
        assert np.allclose(cell_geom.centroid[0], [0.5, 0.5])

    def test_unit_triangle(self):
        # CCW triangle (0,0),(1,0),(0,1): edges (0,1),(1,2),(2,0)
        # Sort parities [0,0,1] -> sgn [-1,-1,+1]
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        face_geom = make_face_geom(coords, [[0, 1], [1, 2], [0, 2]])
        cell_topo = make_cell_topo(
            face_idx=[[0, 1, 2]],
            face_sgn=[[-1, -1, +1]],
        )
        cell_geom = cell_topo.build_geometric(face_geom)
        assert float(cell_geom.volume[0]) == approx(0.5)
        assert np.allclose(cell_geom.centroid[0], [1.0 / 3, 1.0 / 3])

    def test_translated_square(self):
        # Translation invariance: shift by (10, 5)
        coords = np.array([[10.0, 5.0], [11.0, 5.0], [11.0, 6.0], [10.0, 6.0]])
        face_geom = make_face_geom(coords, [[0, 1], [1, 2], [2, 3], [0, 3]])
        cell_topo = make_cell_topo(
            face_idx=[[0, 1, 2, 3]],
            face_sgn=[[-1, -1, -1, +1]],
        )
        cell_geom = cell_topo.build_geometric(face_geom)
        assert float(cell_geom.volume[0]) == approx(1.0)
        assert np.allclose(cell_geom.centroid[0], [10.5, 5.5])

    def test_larger_square(self):
        # 2x2 square at origin: V=4, C=(1,1)
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        face_geom = make_face_geom(coords, [[0, 1], [1, 2], [2, 3], [0, 3]])
        cell_topo = make_cell_topo(
            face_idx=[[0, 1, 2, 3]],
            face_sgn=[[-1, -1, -1, +1]],
        )
        cell_geom = cell_topo.build_geometric(face_geom)
        assert float(cell_geom.volume[0]) == approx(4.0)
        assert np.allclose(cell_geom.centroid[0], [1.0, 1.0])

    def test_two_adjacent_squares_share_a_face(self):
        # Two unit squares side-by-side, sharing edge (1,4)
        # Cell 0 (left, verts 0,1,4,3): edges (0,1),(1,4),(4,3),(3,0)
        # Cell 1 (right, verts 1,2,5,4): edges (1,2),(2,5),(5,4),(4,1)
        coords = np.array([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        ])
        # Unique sorted face keys (in deterministic build order)
        face_keys = [[0, 1], [1, 4], [3, 4], [0, 3], [1, 2], [2, 5], [4, 5]]
        face_geom = make_face_geom(coords, face_keys)
        # Cell 0 face indices into face_keys: bottom=0, right=1, top=2, left=3
        # Cell 0 parities [0,0,1,1] -> sgn [-1,-1,+1,+1]
        # Cell 1: bottom=4, right=5, top=6, left=1 (shared)
        # Cell 1 parities [0,0,1,1] -> sgn [-1,-1,+1,+1]
        cell_topo = make_cell_topo(
            face_idx=[[0, 1, 2, 3], [4, 5, 6, 1]],
            face_sgn=[[-1, -1, +1, +1], [-1, -1, +1, +1]],
        )
        cell_geom = cell_topo.build_geometric(face_geom)
        assert np.allclose(cell_geom.volume, [1.0, 1.0])
        assert np.allclose(cell_geom.centroid[0], [0.5, 0.5])
        assert np.allclose(cell_geom.centroid[1], [1.5, 0.5])


class TestCellGeometry3D:
    def test_unit_tetrahedron(self):
        # Vertices (0,0,0),(1,0,0),(0,1,0),(0,0,1)
        # Face windings: (0,2,1),(0,1,3),(1,2,3),(0,3,2)
        # Sort parities [1,0,0,1] -> sgn [+1,-1,-1,+1]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        face_geom = make_face_geom(coords, [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
        cell_topo = make_cell_topo(
            face_idx=[[0, 1, 2, 3]],
            face_sgn=[[+1, -1, -1, +1]],
        )
        cell_geom = cell_topo.build_geometric(face_geom)
        assert float(cell_geom.volume[0]) == approx(1.0 / 6)
        assert np.allclose(cell_geom.centroid[0], [0.25, 0.25, 0.25])

    def test_translated_tetrahedron(self):
        # Translation invariance: shift unit tet by (1, 2, 3)
        coords = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 3.0],
            [1.0, 3.0, 3.0],
            [1.0, 2.0, 4.0],
        ])
        face_geom = make_face_geom(coords, [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
        cell_topo = make_cell_topo(
            face_idx=[[0, 1, 2, 3]],
            face_sgn=[[+1, -1, -1, +1]],
        )
        cell_geom = cell_topo.build_geometric(face_geom)
        assert float(cell_geom.volume[0]) == approx(1.0 / 6)
        assert np.allclose(cell_geom.centroid[0], [1.25, 2.25, 3.25])

    def test_larger_tetrahedron(self):
        # Scale unit tet by 2: V scales by 2^3 = 8, so V = 8/6 = 4/3
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ])
        face_geom = make_face_geom(coords, [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
        cell_topo = make_cell_topo(
            face_idx=[[0, 1, 2, 3]],
            face_sgn=[[+1, -1, -1, +1]],
        )
        cell_geom = cell_topo.build_geometric(face_geom)
        assert float(cell_geom.volume[0]) == approx(4.0 / 3)
        assert np.allclose(cell_geom.centroid[0], [0.5, 0.5, 0.5])
