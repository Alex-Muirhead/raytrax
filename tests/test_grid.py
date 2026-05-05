import numpy as np
import pytest

from raytrax.grid import sort_with_parity_bit, lex_unique


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
