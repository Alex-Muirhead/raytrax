import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import jax.numpy as jnp
    import meshio
    import numpy as np


@app.cell
def _():
    mesh = meshio.read("../cylinder.msh")
    cell_block = mesh.cells[0]

    cell_vertices = np.asarray(cell_block.data)
    return (cell_block,)


@app.cell(hide_code=True)
def _():
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
            (0, 2, 1),     # bottom triangle, normal pointing down
            (3, 4, 5),     # top triangle, normal pointing up
            (0, 1, 4, 3),  # front quad
            (1, 2, 5, 4),  # right quad
            (2, 0, 3, 5),  # left quad
        ],

        # Pyramid:
        #   Base = 0,1,2,3 (quad), apex = 4 (above).
        "pyramid": [
            (0, 3, 2, 1),  # base quad, normal pointing down (away from apex)
            (0, 1, 4),     # front triangle
            (1, 2, 4),     # right triangle
            (2, 3, 4),     # back triangle
            (3, 0, 4),     # left triangle
        ],
    }
    return (FACE_DEFINITIONS,)


@app.function
def sort3_with_parity(arr):
    """Sorting network for 3 elements, returns (sorted, parity).

    Optimal: 3 comparisons.
    """
    a, b, c = arr[0], arr[1], arr[2]
    swaps = 0

    def cmp_swap(x, y, s):
        need_swap = x > y
        lo = jnp.where(need_swap, y, x)
        hi = jnp.where(need_swap, x, y)
        return lo, hi, s + jnp.where(need_swap, 1, 0)

    a, b, swaps = cmp_swap(a, b, swaps)
    b, c, swaps = cmp_swap(b, c, swaps)
    a, b, swaps = cmp_swap(a, b, swaps)

    sorted_arr = jnp.stack([a, b, c])
    parity = jnp.where(swaps % 2 == 0, 1, -1)
    return sorted_arr, parity


@app.cell
def _(FACE_DEFINITIONS: dict[str, list[tuple[int, ...]]], cell_block):
    cell_face_vertices = np.asarray(cell_block.data[:, FACE_DEFINITIONS[cell_block.type]], dtype=jnp.int32)
    cell_face_vertices = np.reshape(cell_face_vertices, (-1, 3))  # 3 <- Number of vertices per face
    cell_face_lexkeys, cell_face_parity = jax.vmap(sort3_with_parity)(cell_face_vertices)
    return cell_face_lexkeys, cell_face_parity


@app.cell
def _(cell_block, cell_face_lexkeys, cell_face_parity):
    cell_ids, local_face_ids = np.meshgrid(np.arange(len(cell_block)), range(4), indexing="ij")
    cell_ids = np.reshape(cell_ids, (-1))
    local_face_ids = np.reshape(local_face_ids, (-1))

    # Sorting BLOCK (idx1, idx2, idx3, parity)
    face_fullkeys = np.column_stack((cell_face_lexkeys, cell_face_parity))
    face_lexkeys_order = np.lexsort(face_fullkeys[..., ::-1].T)
    # The order will have parity=-1 then parity=+1

    # Sort all our data (so we can keep track of it)
    sorted_face_lexkeys = cell_face_lexkeys[face_lexkeys_order, :]
    sorted_face_parity = cell_face_parity[face_lexkeys_order]
    sorted_cell_ids = cell_ids[face_lexkeys_order]
    sorted_local_face_ids = local_face_ids[face_lexkeys_order]
    return sorted_face_lexkeys, sorted_face_parity


@app.cell
def _(sorted_face_lexkeys):
    # Assumption -> Faces (or their lexkeys) appear AT MOST twice.
    unique_face_lexkeys, unique_index, sorted_face_ids, face_lexkey_counts = np.unique(
        sorted_face_lexkeys,
        return_index=True,
        return_inverse=True, 
        return_counts=True, 
        sorted=True, 
        axis=0
    )
    assert np.max(face_lexkey_counts) <= 2, "Invalid mesh: Faces appear connected to more than 2 cells"
    return (sorted_face_ids,)


@app.cell
def _(sorted_face_lexkeys, sorted_face_parity):
    negative_parity_pairing = np.all(np.roll(sorted_face_lexkeys, -1, axis=0) == sorted_face_lexkeys, axis=-1)
    positive_parity_pairing = np.all(np.roll(sorted_face_lexkeys, +1, axis=0) == sorted_face_lexkeys, axis=-1)

    # What (duplicated) faces are connected? 
    # As the sorted_... arrays are also sorted by parity, matched_idx should be parity=-1, matched_idx+1 should be +1
    assert np.all(sorted_face_parity[negative_parity_pairing] == -1), "Uh-oh, parity is off!"
    assert np.all(sorted_face_parity[positive_parity_pairing] == +1), "Uh-oh, parity is off!"
    return negative_parity_pairing, positive_parity_pairing


@app.cell
def _(negative_parity_pairing, positive_parity_pairing):
    print(negative_parity_pairing[:20])
    print(positive_parity_pairing[:20])
    return


@app.cell
def _(sorted_face_ids):
    sorted_face_ids
    return


if __name__ == "__main__":
    app.run()
