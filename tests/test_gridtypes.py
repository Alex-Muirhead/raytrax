import numpy as np
import pytest
from pytest import approx

from raytrax.gridtypes import FaceTopology


def make_topo(vertex_indices):
    return FaceTopology(
        vertices=np.array(vertex_indices),
        cells=np.zeros(0, dtype=int),
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
