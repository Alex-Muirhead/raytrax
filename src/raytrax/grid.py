import numpy as np

FACE_DEFINITIONS: dict[str, list[tuple[int, ...]]] = {
    # 2D elements: "faces" are edges.
    # Winding is counterclockwise, so the outward normal points
    # to the right of the edge direction (i.e., outward from the element).
    "triangle": [
        (0, 1),  # bottom edge
        (1, 2),  # right edge
        (2, 0),  # left edge
    ],
    "quad": [
        (0, 1),  # bottom: 0 -> 1
        (1, 2),  # right:  1 -> 2
        (2, 3),  # top:    2 -> 3
        (3, 0),  # left:   3 -> 0
    ],
    # 3D elements: faces wound counterclockwise when viewed from outside.
    #
    # Tetrahedron:
    #   Base = 0,1,2 (z=0 plane), apex = 3 (above).
    #   Base normal points downward (away from 3) -> wind CW from above = CCW from below.
    "tetra": [
        (0, 2, 1),  # base face, normal pointing down (away from vertex 3)
        (0, 1, 3),  # front face, normal pointing away from vertex 2
        (1, 2, 3),  # right face, normal pointing away from vertex 0
        (0, 3, 2),  # left face, normal pointing away from vertex 1
    ],
    # Hexahedron:
    #   Bottom = 0,1,2,3 (CCW from below), Top = 4,5,6,7 (CCW from above).
    #   Vertex i on bottom connects to vertex i+4 on top.
    "hexahedron": [
        (0, 3, 2, 1),  # bottom face, normal pointing down
        (4, 5, 6, 7),  # top face, normal pointing up
        (0, 1, 5, 4),  # front face, normal pointing out
        (1, 2, 6, 5),  # right face, normal pointing out
        (2, 3, 7, 6),  # back face, normal pointing out
        (3, 0, 4, 7),  # left face, normal pointing out
    ],
    # Wedge (triangular prism):
    #   Bottom triangle = 0,1,2, Top triangle = 3,4,5.
    #   Vertex i on bottom connects to vertex i+3 on top.
    "wedge": [
        (0, 2, 1),  # bottom triangle, normal pointing down
        (3, 4, 5),  # top triangle, normal pointing up
        (0, 1, 4, 3),  # front quad
        (1, 2, 5, 4),  # right quad
        (2, 0, 3, 5),  # left quad
    ],
    # Pyramid:
    #   Base = 0,1,2,3 (quad), apex = 4 (above).
    "pyramid": [
        (0, 3, 2, 1),  # base quad, normal pointing down (away from apex)
        (0, 1, 4),  # front triangle
        (1, 2, 4),  # right triangle
        (2, 3, 4),  # back triangle
        (3, 0, 4),  # left triangle
    ],
}


def sort3_with_parity_bit(arr, *, axis: int = 0):
    """Sorting for 3 elements, returns (sorted, parity)."""
    a, b, c = np.unstack(arr, axis=axis)
    swaps = 0

    def cmp_swap(x, y, s):
        need_swap = x > y
        lo = np.where(need_swap, y, x)
        hi = np.where(need_swap, x, y)
        return lo, hi, s + np.where(need_swap, 1, 0)

    a, b, swaps = cmp_swap(a, b, swaps)
    b, c, swaps = cmp_swap(b, c, swaps)
    a, b, swaps = cmp_swap(a, b, swaps)

    sorted_arr = np.stack([a, b, c], axis=axis)
    parity_bit = swaps % 2
    return sorted_arr, parity_bit


def plane_normal_and_offset(arr, *, axis: int = 0, np=np):
    """Defining a plane from 3 points, returns (normal, offset)."""
    ref, u, v = np.unstack(arr, axis=axis)
    # Relative vectors
    u -= ref
    v -= ref
    normal = np.cross(u, v)
    normal /= np.linalg.vector_norm(normal)  # Ensure unit vec
    offset = np.dot(ref, normal)
    return normal, offset


def rearrange(array, *, from_):
    out = np.empty_like(array)
    out[from_] = array
    return out


def lex_unique(
    keys,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    """A stripped-down version of numpy.unique that uses lexsort on keys."""
    if len(keys.shape) != 2:
        raise ValueError("Keys must be 2 dimensional")

    sorting_order = np.lexsort(np.unstack(keys, axis=1))

    # <--- Start region of "ordered faces"
    sorted_keys = keys[sorting_order, :]
    # If we represent "sorted_keys" as letters (for simplicity).
    #
    #          sorted_keys      :: a, b, b, c, c, d, e, f, f, g, ..., z, z
    #  np.roll(sorted_keys, +1) :: z, a, b, b, c, c, d, e, f, f, g, ..., z
    # ------------------------- :: ---------------------------------------
    #                 mask      :: T, T, F, T, F, T, T, T, F, T,    ..., F
    #     sorted_keys[mask]     :: a, b, c, d, e, f, g, ...
    #       sorted_key_ids      :: 0, 1, 1, 2, 2, 3, 4, 5, 5, 6, ..., n, n
    #
    # We guarantee that the first face_key must be the first occurance by ordering.
    # Therefore, the first face_id is guaranteed to be 0.
    is_first_instance = np.ones(sorting_order.size, dtype=bool)
    is_first_instance[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=-1)
    ret = (sorted_keys[is_first_instance],)

    if return_index:
        ret += (sorting_order[is_first_instance],)
    if return_inverse:
        sorted_key_ids = np.cumulative_sum(is_first_instance) - 1  # Accumulative version of count_nonzero
        key_ids = np.empty_like(sorting_order)
        key_ids[sorting_order] = sorted_key_ids
        ret += (key_ids,)
    if return_counts:
        idx = np.concat(np.nonzero(is_first_instance) + ([is_first_instance.size],))
        ret += (np.diff(idx),)

    return ret


def process_cell_block(cell_block, *, debug: bool = False):
    """TODO."""
    cell_face_structure = FACE_DEFINITIONS[cell_block.type]

    num_cells = len(cell_block)
    num_faces_per_cell = len(cell_face_structure)
    num_verts_per_face, *other = set(len(ids) for ids in cell_face_structure)
    if other:
        raise ValueError("Cannot handle meshes with variable faces per cell")

    # Prefixes!
    #  - `cell_` has first axis indexing cell
    #  - `cell_face_` has first axis indexing cell, second indexing face
    #  - `all_face_` has duplicate faces (can be reshaped to `cell_face_`)
    #  - `face_` has first axis indexing face

    cell_face_vertices = cell_block.data[:, cell_face_structure]

    # V-mapping makes this muuuch faster
    _cell_face_keys, cell_face_paritybit = sort3_with_parity_bit(cell_face_vertices, axis=2)

    if debug:
        print("Number of faces that are referenced with parity")
        print(f"\t even: {np.count_nonzero(cell_face_paritybit == 0):10,}")
        print(f"\t  odd: {np.count_nonzero(cell_face_paritybit == 1):10,}")

        _, face_lexkey_counts = np.unique(_cell_face_keys, return_counts=True, axis=0)
        assert np.max(face_lexkey_counts) <= 2, "Invalid mesh: Faces appear connected to more than 2 cells"

    # Find the unique faces (efficiently)
    _all_face_keys = _cell_face_keys.reshape((-1, num_verts_per_face))
    _face_keys, all_face_ids = lex_unique(_all_face_keys, return_inverse=True)

    cell_face_ids = all_face_ids.reshape((num_cells, num_faces_per_cell))

    cell_ids = np.expand_dims(range(num_cells), axis=1)  # Column vector
    num_faces, _ = _face_keys.shape

    # WARN: We are using a sentinal value of -1 here!
    face_cell_ids = np.full((num_faces, 2), fill_value=-1, dtype=int)
    face_cell_ids[cell_face_ids, cell_face_paritybit] = cell_ids

    if debug:
        assert np.all(face_cell_ids[cell_face_ids, cell_face_paritybit] == cell_ids), "Indexing is messed up"

    # Now we reconstruct adjacency!
    # WARN: We are using a sentinal value of -1 here!
    cell_to_cell = face_cell_ids[cell_face_ids, 1 - cell_face_paritybit]

    return cell_to_cell, face_cell_ids
