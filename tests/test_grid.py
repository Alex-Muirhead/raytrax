import meshio
import numpy as np
import pytest
from pytest import approx

from raytrax.grid import lex_unique, process_cell_block, sort_with_parity_bit


class TestSortWithParityBit:
    # --- length-2 ---

    @pytest.mark.parametrize(
        "perm, expected_parity",
        [
            ([1, 2], 0),  # already sorted
            ([2, 1], 1),  # one swap
        ],
    )
    def test_all_permutations_of_two(self, perm, expected_parity):
        arr = np.array(perm)
        sorted_arr, parity = sort_with_parity_bit(arr, axis=0)
        assert np.array_equal(sorted_arr, np.array([1, 2]))
        assert int(parity) == expected_parity

    def test_equal_elements_two(self):
        arr = np.array([3, 3])
        sorted_arr, parity = sort_with_parity_bit(arr, axis=0)
        assert np.array_equal(sorted_arr, np.array([3, 3]))
        assert int(parity) == 0

    def test_batched_two_axis0(self):
        # Shape (2, N): each column is an independent pair to sort.
        # Column 0: (2, 1) -> [1, 2], parity 1
        # Column 1: (1, 2) -> [1, 2], parity 0
        arr = np.array([[2, 1], [1, 2]])
        sorted_arr, parity = sort_with_parity_bit(arr, axis=0)
        assert np.array_equal(sorted_arr, np.array([[1, 1], [2, 2]]))
        assert np.array_equal(parity, np.array([1, 0]))

    def test_batched_two_axis1(self):
        # Shape (N, 2): each row is an independent pair to sort.
        # Row 0: (2, 1) -> [1, 2], parity 1
        # Row 1: (1, 2) -> [1, 2], parity 0
        arr = np.array([[2, 1], [1, 2]])
        sorted_arr, parity = sort_with_parity_bit(arr, axis=1)
        assert np.array_equal(sorted_arr, np.array([[1, 2], [1, 2]]))
        assert np.array_equal(parity, np.array([1, 0]))

    # --- length-3 ---

    @pytest.mark.parametrize(
        "perm, expected_parity",
        [
            ([1, 2, 3], 0),  # identity permutation
            ([1, 3, 2], 1),
            ([2, 1, 3], 1),
            ([2, 3, 1], 0),
            ([3, 1, 2], 0),
            ([3, 2, 1], 1),
        ],
    )
    def test_all_permutations_of_three(self, perm, expected_parity):
        arr = np.array(perm)
        sorted_arr, parity = sort_with_parity_bit(arr, axis=0)
        assert np.array_equal(sorted_arr, np.array([1, 2, 3]))
        assert int(parity) == expected_parity

    def test_equal_adjacent_elements(self):
        arr = np.array([2, 2, 1])
        sorted_arr, _ = sort_with_parity_bit(arr, axis=0)
        assert np.array_equal(sorted_arr, np.array([1, 2, 2]))

    def test_all_equal_elements(self):
        arr = np.array([5, 5, 5])
        sorted_arr, parity = sort_with_parity_bit(arr, axis=0)
        assert np.array_equal(sorted_arr, np.array([5, 5, 5]))
        assert int(parity) == 0

    def test_batched_axis0(self):
        # Shape (3, N): each column is an independent triplet to sort.
        # Column 0: (3, 1, 2) -> [1,2,3], parity 0
        # Column 1: (1, 2, 3) -> [1,2,3], parity 0
        # Column 2: (2, 3, 1) -> [1,2,3], parity 0
        arr = np.array([[3, 1, 2], [1, 2, 3], [2, 3, 1]])
        sorted_arr, parity = sort_with_parity_bit(arr, axis=0)
        assert np.array_equal(sorted_arr, np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]))
        assert np.array_equal(parity, np.array([0, 0, 0]))

    def test_batched_axis1(self):
        # Shape (N, 3): each row is an independent triplet to sort.
        # Row 0: (3, 1, 2) -> [1,2,3], parity 0
        # Row 1: (1, 3, 2) -> [1,2,3], parity 1
        arr = np.array([[3, 1, 2], [1, 3, 2]])
        sorted_arr, parity = sort_with_parity_bit(arr, axis=1)
        assert np.array_equal(sorted_arr, np.array([[1, 2, 3], [1, 2, 3]]))
        assert np.array_equal(parity, np.array([0, 1]))

    # --- length-4 ---

    @pytest.mark.parametrize(
        "perm, expected_parity",
        [
            ([1, 2, 3, 4], 0),  # identity
            ([1, 2, 4, 3], 1),
            ([1, 3, 2, 4], 1),
            ([1, 3, 4, 2], 0),
            ([1, 4, 2, 3], 0),
            ([1, 4, 3, 2], 1),
            ([2, 1, 3, 4], 1),
            ([2, 1, 4, 3], 0),
            ([2, 3, 1, 4], 0),
            ([2, 3, 4, 1], 1),
            ([2, 4, 1, 3], 1),
            ([2, 4, 3, 1], 0),
            ([3, 1, 2, 4], 0),
            ([3, 1, 4, 2], 1),
            ([3, 2, 1, 4], 1),
            ([3, 2, 4, 1], 0),
            ([3, 4, 1, 2], 0),
            ([3, 4, 2, 1], 1),
            ([4, 1, 2, 3], 1),
            ([4, 1, 3, 2], 0),
            ([4, 2, 1, 3], 0),
            ([4, 2, 3, 1], 1),
            ([4, 3, 1, 2], 1),
            ([4, 3, 2, 1], 0),
        ],
    )
    def test_all_permutations_of_four(self, perm, expected_parity):
        arr = np.array(perm)
        sorted_arr, parity = sort_with_parity_bit(arr, axis=0)
        assert np.array_equal(sorted_arr, np.array([1, 2, 3, 4]))
        assert int(parity) == expected_parity

    def test_equal_elements_four(self):
        arr = np.array([3, 1, 3, 1])
        sorted_arr, _ = sort_with_parity_bit(arr, axis=0)
        assert np.array_equal(sorted_arr, np.array([1, 1, 3, 3]))

    def test_all_equal_elements_four(self):
        arr = np.array([7, 7, 7, 7])
        sorted_arr, parity = sort_with_parity_bit(arr, axis=0)
        assert np.array_equal(sorted_arr, np.array([7, 7, 7, 7]))
        assert int(parity) == 0

    def test_batched_four_axis1(self):
        # Shape (N, 4): each row is an independent quadruple to sort.
        # Row 0: (4,3,2,1) -> [1,2,3,4], parity 0
        # Row 1: (1,4,3,2) -> [1,2,3,4], parity 1
        arr = np.array([[4, 3, 2, 1], [1, 4, 3, 2]])
        sorted_arr, parity = sort_with_parity_bit(arr, axis=1)
        assert np.array_equal(sorted_arr, np.array([[1, 2, 3, 4], [1, 2, 3, 4]]))
        assert np.array_equal(parity, np.array([0, 1]))

    def test_parity_is_binary(self):
        for perm in [[1, 2], [2, 1], [1, 2, 3], [3, 2, 1], [2, 1, 3], [4, 3, 2, 1], [1, 4, 3, 2]]:
            _, parity = sort_with_parity_bit(np.array(perm), axis=0)
            assert int(parity) in (0, 1)

    # --- error cases ---

    def test_raises_for_unsupported_length(self):
        arr = np.array([1, 2, 3, 4, 5])
        with pytest.raises(NotImplementedError):
            sort_with_parity_bit(arr, axis=0)


class TestLexUnique:
    def test_basic_deduplication(self):
        keys = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
        (unique,) = lex_unique(keys)
        assert unique.shape == (3, 2)
        assert _rows_equal_as_sets(unique, np.array([[1, 2], [3, 4], [5, 6]]))

    def test_already_unique(self):
        keys = np.array([[1, 0], [2, 0], [3, 0]])
        (unique,) = lex_unique(keys)
        assert unique.shape == (3, 2)
        assert _rows_equal_as_sets(unique, keys)

    def test_all_duplicates(self):
        keys = np.array([[7, 8], [7, 8], [7, 8]])
        (unique,) = lex_unique(keys)
        assert np.array_equal(unique, np.array([[7, 8]]))

    def test_return_index_points_to_first_occurrence(self):
        keys = np.array([[3, 4], [1, 2], [1, 2], [3, 4]])
        unique, idx = lex_unique(keys, return_index=True)
        assert np.array_equal(keys[idx], unique)

    def test_return_inverse_reconstructs_input(self):
        keys = np.array([[1, 2], [3, 4], [1, 2], [5, 6], [3, 4]])
        unique, inv = lex_unique(keys, return_inverse=True)
        assert np.array_equal(unique[inv], keys)

    def test_return_counts_sum_to_input_length(self):
        keys = np.array([[1, 2], [3, 4], [1, 2], [1, 2], [3, 4]])
        unique, counts = lex_unique(keys, return_counts=True)
        assert np.sum(counts) == len(keys)
        assert len(counts) == len(unique)
        assert np.all(counts > 0)

    def test_return_counts_values(self):
        keys = np.array([[1, 2], [1, 2], [3, 4]])
        unique, counts = lex_unique(keys, return_counts=True)
        count_for_row = {tuple(unique[i]): counts[i] for i in range(len(unique))}
        assert count_for_row[(1, 2)] == 2
        assert count_for_row[(3, 4)] == 1

    def test_all_return_flags_consistent(self):
        keys = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
        unique, idx, inv, counts = lex_unique(keys, return_index=True, return_inverse=True, return_counts=True)
        assert np.array_equal(keys[idx], unique)
        assert np.array_equal(unique[inv], keys)
        assert np.sum(counts) == len(keys)
        assert len(counts) == len(unique)

    def test_raises_on_1d_input(self):
        with pytest.raises(ValueError, match="2 dimensional"):
            lex_unique(np.array([1, 2, 3]))

    def test_raises_on_3d_input(self):
        with pytest.raises(ValueError):
            lex_unique(np.array([[[1, 2], [3, 4]]]))

    def test_single_column_keys(self):
        keys = np.array([[3], [1], [1], [2]])
        unique, inv = lex_unique(keys, return_inverse=True)
        assert np.array_equal(unique[inv], keys)
        assert unique.shape == (3, 1)

    def test_single_row_input(self):
        keys = np.array([[4, 5]])
        (unique,) = lex_unique(keys)
        assert np.array_equal(unique, np.array([[4, 5]]))


def _rows_equal_as_sets(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a and b contain the same rows (order-independent)."""
    a_sorted = a[np.lexsort(a.T[::-1])]
    b_sorted = b[np.lexsort(b.T[::-1])]
    return np.array_equal(a_sorted, b_sorted)


def _make_cell_block(cell_type: str, cells):
    return meshio.CellBlock(cell_type, np.asarray(cells))


def _find_face_row(face_keys: np.ndarray, vertices) -> int:
    """Return the row index in face_keys whose entries (as a set) match `vertices`."""
    target = set(vertices)
    matches = [i for i, row in enumerate(face_keys) if set(row.tolist()) == target]
    assert len(matches) == 1, f"expected exactly one face matching {vertices}, found {matches}"
    return matches[0]


class TestProcessCellBlockTriangle:
    def test_single_triangle_topology(self):
        block = _make_cell_block("triangle", [[0, 1, 2]])
        topo = process_cell_block(block)

        # Three unique edges, each with two vertices
        assert topo.faces.vertices.shape == (3, 2)
        # Every face is on the boundary: exactly one cell + one sentinel
        boundary_count = (topo.faces.cells == -1).sum(axis=1)
        assert np.all(boundary_count == 1)
        # The lone cell appears once per face
        assert (topo.faces.cells == 0).sum() == 3
        # Cell has three faces, all boundary
        assert topo.cells.face_ids.shape == (1, 3)
        assert topo.cells.face_signs.shape == (1, 3)
        assert np.all(np.isin(topo.cells.face_signs, [-1, +1]))
        assert np.all(topo.cells.neighbours == -1)

    def test_two_adjacent_triangles_share_a_face(self):
        # Quad split along diagonal (0,2)
        block = _make_cell_block("triangle", [[0, 1, 2], [0, 2, 3]])
        topo = process_cell_block(block)

        # Five unique edges: 3 + 3 - 1 shared
        assert topo.faces.vertices.shape == (5, 2)

        # Exactly one interior face (with both cells assigned)
        boundary_count = (topo.faces.cells == -1).sum(axis=1)
        assert (boundary_count == 0).sum() == 1
        assert (boundary_count == 1).sum() == 4

        # The shared edge is (0,2), and it connects cells 0 and 1
        shared = _find_face_row(topo.faces.vertices, [0, 2])
        assert set(topo.faces.cells[shared].tolist()) == {0, 1}

        # Adjacency: each cell has exactly one non-boundary neighbour, the other cell
        cell0_neighbours = set(topo.cells.neighbours[0].tolist()) - {-1}
        cell1_neighbours = set(topo.cells.neighbours[1].tolist()) - {-1}
        assert cell0_neighbours == {1}
        assert cell1_neighbours == {0}


class TestProcessCellBlockQuad:
    def test_single_quad_topology(self):
        block = _make_cell_block("quad", [[0, 1, 2, 3]])
        topo = process_cell_block(block)

        assert topo.faces.vertices.shape == (4, 2)
        boundary_count = (topo.faces.cells == -1).sum(axis=1)
        assert np.all(boundary_count == 1)
        assert topo.cells.face_ids.shape == (1, 4)
        assert np.all(topo.cells.neighbours == -1)


class TestProcessCellBlockTetra:
    def test_single_tetrahedron_topology(self):
        block = _make_cell_block("tetra", [[0, 1, 2, 3]])
        topo = process_cell_block(block)

        # Four triangular faces, all boundary
        assert topo.faces.vertices.shape == (4, 3)
        boundary_count = (topo.faces.cells == -1).sum(axis=1)
        assert np.all(boundary_count == 1)
        assert topo.cells.face_ids.shape == (1, 4)
        assert np.all(topo.cells.neighbours == -1)

    def test_two_adjacent_tetrahedra_share_a_face(self):
        # Tet 0 fills the corner near origin; tet 1 has apex (1,1,1)
        # and winds opposite (1,3,2) so the shared face has consistent orientation.
        block = _make_cell_block("tetra", [[0, 1, 2, 3], [4, 1, 3, 2]])
        topo = process_cell_block(block)

        # 4 + 4 - 1 = 7 unique faces
        assert topo.faces.vertices.shape == (7, 3)

        # Exactly one interior face
        boundary_count = (topo.faces.cells == -1).sum(axis=1)
        assert (boundary_count == 0).sum() == 1

        # Shared face is the triangle (1,2,3)
        shared = _find_face_row(topo.faces.vertices, [1, 2, 3])
        assert set(topo.faces.cells[shared].tolist()) == {0, 1}

        cell0_neighbours = set(topo.cells.neighbours[0].tolist()) - {-1}
        cell1_neighbours = set(topo.cells.neighbours[1].tolist()) - {-1}
        assert cell0_neighbours == {1}
        assert cell1_neighbours == {0}


class TestProcessCellBlockGeometry:
    """End-to-end: process_cell_block + MeshTopology.build_geometric."""

    def test_unit_triangle(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        block = _make_cell_block("triangle", [[0, 1, 2]])
        geom = process_cell_block(block).build_geometric(points)

        assert float(geom.cells.volume[0]) == approx(0.5)
        assert np.allclose(geom.cells.centroid[0], [1.0 / 3, 1.0 / 3])

    def test_unit_quad(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        block = _make_cell_block("quad", [[0, 1, 2, 3]])
        geom = process_cell_block(block).build_geometric(points)

        assert float(geom.cells.volume[0]) == approx(1.0)
        assert np.allclose(geom.cells.centroid[0], [0.5, 0.5])

    def test_unit_tetrahedron(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        block = _make_cell_block("tetra", [[0, 1, 2, 3]])
        geom = process_cell_block(block).build_geometric(points)

        assert float(geom.cells.volume[0]) == approx(1.0 / 6)
        assert np.allclose(geom.cells.centroid[0], [0.25, 0.25, 0.25])

    def test_two_triangles_split_unit_square(self):
        # Diagonal split: each triangle has area 1/2
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        block = _make_cell_block("triangle", [[0, 1, 2], [0, 2, 3]])
        geom = process_cell_block(block).build_geometric(points)

        assert np.allclose(geom.cells.volume, [0.5, 0.5])
        # Centroid of triangle (0,0),(1,0),(1,1) is (2/3, 1/3)
        # Centroid of triangle (0,0),(1,1),(0,1) is (1/3, 2/3)
        assert np.allclose(geom.cells.centroid[0], [2.0 / 3, 1.0 / 3])
        assert np.allclose(geom.cells.centroid[1], [1.0 / 3, 2.0 / 3])

    def test_two_tetrahedra(self):
        # Tet 0 (corner at origin) + tet 1 (apex at (1,1,1))
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        block = _make_cell_block("tetra", [[0, 1, 2, 3], [4, 1, 3, 2]])
        geom = process_cell_block(block).build_geometric(points)

        # Corner tet: V = 1/6, centroid at (1/4, 1/4, 1/4)
        assert float(geom.cells.volume[0]) == approx(1.0 / 6)
        assert np.allclose(geom.cells.centroid[0], [0.25, 0.25, 0.25])
        # Apex tet: V = 1/3 (mean of 4 vertices = (0.5, 0.5, 0.5))
        assert float(geom.cells.volume[1]) == approx(1.0 / 3)
        assert np.allclose(geom.cells.centroid[1], [0.5, 0.5, 0.5])

    def test_translated_triangle(self):
        # Translation invariance: shift unit triangle by (10, 5)
        points = np.array([[10.0, 5.0], [11.0, 5.0], [10.0, 6.0]])
        block = _make_cell_block("triangle", [[0, 1, 2]])
        geom = process_cell_block(block).build_geometric(points)

        assert float(geom.cells.volume[0]) == approx(0.5)
        assert np.allclose(geom.cells.centroid[0], [10.0 + 1.0 / 3, 5.0 + 1.0 / 3])
