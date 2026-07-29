import equinox as eqx
import numpy as np

from raytrax.gridtypes import CellTopology, FaceTopology, MeshTopology

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


def sort_with_parity_bit(arr, *, axis: int = -2):
    """Sorting for 2, 3 and 4 elements, returns (sorted, parity)."""
    swaps = 0

    def cmp_swap(x, y, s):
        need_swap = x > y
        lo = np.where(need_swap, y, x)
        hi = np.where(need_swap, x, y)
        return lo, hi, s + np.where(need_swap, 1, 0)

    match arr.shape[axis]:
        case 2:
            a, b = np.unstack(arr, axis=axis)
            a, b, swaps = cmp_swap(a, b, swaps)
            sorted_arr = np.stack([a, b], axis=axis)

        case 3:
            a, b, c = np.unstack(arr, axis=axis)

            a, b, swaps = cmp_swap(a, b, swaps)
            b, c, swaps = cmp_swap(b, c, swaps)
            a, b, swaps = cmp_swap(a, b, swaps)

            sorted_arr = np.stack([a, b, c], axis=axis)

        case 4:
            # Bose-Nelson 5-comparator sorting network:
            # (a,b)(c,d) -> (a,c)(b,d) -> (b,c)
            a, b, c, d = np.unstack(arr, axis=axis)

            a, b, swaps = cmp_swap(a, b, swaps)
            c, d, swaps = cmp_swap(c, d, swaps)
            a, c, swaps = cmp_swap(a, c, swaps)
            b, d, swaps = cmp_swap(b, d, swaps)
            b, c, swaps = cmp_swap(b, c, swaps)

            sorted_arr = np.stack([a, b, c, d], axis=axis)

        case _:
            raise NotImplementedError("Sorting only implemented for lengths 2, 3, and 4")

    parity_bit = swaps % 2
    return sorted_arr, parity_bit


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


def match_faces(face_topology: FaceTopology, face_vertices) -> np.ndarray:
    """Find the face ids of faces given by their (unsorted) vertex ids."""
    keys, _ = sort_with_parity_bit(np.asarray(face_vertices), axis=-1)
    stored = np.asarray(face_topology.vertices)
    num_stored, _ = stored.shape

    # If every query key already exists in `stored`, the unique keys of the
    # combined array are exactly the stored keys.
    combined_keys, combined_ids = lex_unique(np.concatenate([stored, keys], axis=0), return_inverse=True)
    if len(combined_keys) != num_stored:
        raise ValueError("Some queried faces do not exist in the mesh")

    unique_to_stored = np.empty(num_stored, dtype=int)
    unique_to_stored[combined_ids[:num_stored]] = np.arange(num_stored)
    return unique_to_stored[combined_ids[num_stored:]]


def boundary_groups(mesh_topology: MeshTopology, mesh) -> dict[str, np.ndarray]:
    """Map tagged face groups of a meshio mesh to boundary face ids."""
    face_types = {2: "line", 3: "triangle", 4: "quad"}
    face_type = face_types[mesh_topology.faces.vertices.shape[1]]
    all_faces = np.concatenate([block.data for block in mesh.cells if block.type == face_type])
    face_cells = np.asarray(mesh_topology.faces.cells)

    groups = {}
    for name, sets in mesh.cell_sets_dict.items():
        if name == "gmsh:bounding_entities" or face_type not in sets:
            continue
        face_ids = match_faces(mesh_topology.faces, all_faces[sets[face_type]])
        if np.any(np.all(face_cells[face_ids] != -1, axis=-1)):
            raise ValueError(f"Face group {name!r} contains interior faces")
        groups[name] = face_ids
    return groups


def apply_boundaries(mesh_topology: MeshTopology, groups: dict[str, np.ndarray]) -> tuple[MeshTopology, dict[str, int]]:
    """Assign a distinct negative sentinel to each boundary face group.

    Sentinels are written into `cells.neighbours`; -1 stays reserved for
    untagged boundaries. Returns the updated topology and the name -> sentinel
    mapping.
    """
    neighbours = np.asarray(mesh_topology.cells.neighbours).copy()
    cell_face_ids = np.asarray(mesh_topology.cells.face_ids)
    face_cells = np.asarray(mesh_topology.faces.cells)

    sentinels = {}
    for offset, (name, face_ids) in enumerate(groups.items()):
        sentinel = -(offset + 2)
        interior_cells = face_cells[face_ids].max(axis=-1)
        local_slots = np.argmax(cell_face_ids[interior_cells] == face_ids[:, None], axis=-1)
        if np.any(neighbours[interior_cells, local_slots] != -1):
            raise ValueError(f"Face group {name!r} contains interior or already-tagged faces")
        neighbours[interior_cells, local_slots] = sentinel
        sentinels[name] = sentinel

    mesh_topology = eqx.tree_at(lambda t: t.cells.neighbours, mesh_topology, neighbours)
    return mesh_topology, sentinels


def process_cell_block(cell_block) -> MeshTopology:
    """Build a `MeshTopology` from a meshio `CellBlock`.

    Supports cell types where every face has the same number of vertices:
    triangles, quads, tetrahedra, and hexahedra. Boundary faces use a
    sentinel of -1 for the missing adjacent cell.
    """
    cell_face_structure = FACE_DEFINITIONS[cell_block.type]

    num_cells = len(cell_block)
    num_faces_per_cell = len(cell_face_structure)
    num_verts_per_face, *other = set(len(ids) for ids in cell_face_structure)
    if other:
        raise ValueError("Cannot handle meshes with variable vertices per face")

    # Prefixes!
    #  - `cell_` has first axis indexing cell
    #  - `cell_face_` has first axis indexing cell, second indexing face
    #  - `all_face_` has duplicate faces (can be reshaped to `cell_face_`)
    #  - `face_` has first axis indexing face

    cell_vertices = cell_block.data
    cell_face_vertices = cell_block.data[:, cell_face_structure]

    # --- Step 1. Discover the faces ---

    cell_face_keys, cell_face_parity = sort_with_parity_bit(cell_face_vertices, axis=2)

    # Find the unique faces (efficiently)
    all_face_keys = cell_face_keys.reshape((-1, num_verts_per_face))
    face_keys, all_face_ids, face_counts = lex_unique(all_face_keys, return_inverse=True, return_counts=True)
    assert np.max(face_counts) <= 2, "Invalid mesh: Faces appear connected to more than 2 cells"

    cell_face_ids = all_face_ids.reshape((num_cells, num_faces_per_cell))

    # --- Step 2. Assemble topology information ---
    # Knit together two-way information. (Cell -> Face) & (Face -> Cell)

    cell_ids = np.expand_dims(range(num_cells), axis=1)  # Column vector

    # WARN: We are using a sentinal value of -1 here!
    num_faces, _ = face_keys.shape
    face_cell_ids = np.full((num_faces, 2), fill_value=-1, dtype=int)
    face_cell_ids[cell_face_ids, cell_face_parity] = cell_ids
    assert np.all(face_cell_ids[cell_face_ids, cell_face_parity] == cell_ids), "Indexing is messed up"

    face_topology = FaceTopology(vertices=face_keys, cells=face_cell_ids)

    # Now we reconstruct adjacency!
    # WARN: We are using a sentinal value of -1 here!
    cell_to_cell = face_cell_ids[cell_face_ids, 1 - cell_face_parity]
    cell_face_signs = np.where(cell_face_parity == 0, +1, -1)

    cell_topology = CellTopology(
        face_ids=cell_face_ids,
        face_signs=cell_face_signs,
        vertices=cell_vertices,
        neighbours=cell_to_cell,
    )

    return MeshTopology(faces=face_topology, cells=cell_topology)
