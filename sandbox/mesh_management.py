import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import meshio
    import numpy as np


@app.cell
def _():
    mesh = meshio.read("cylinder.msh")
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
def sort3_with_parity_bit(arr):
    """Sorting for 3 elements, returns (sorted, parity)."""
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
    parity = swaps % 2
    return sorted_arr, parity


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This is where things get tricky to keep track of.
    I've flattened everything, but all of these are "grouped" by the cell.
    """)
    return


@app.cell
def _(cell_block):
    num_cells = len(cell_block)
    num_faces_per_cell = 4  # Hard-code for now
    num_verts_per_face = 3  # Hard-code for now
    return num_cells, num_faces_per_cell, num_verts_per_face


@app.cell
def _(
    FACE_DEFINITIONS: dict[str, list[tuple[int, ...]]],
    cell_block,
    num_cells,
    num_faces_per_cell,
    num_verts_per_face,
):
    # As we perform operations on the face lists, we want to keep track of where they came from.
    # We create an array of shape (num_cells, num_faces) that enumerates the cells & (local) faces. 
    cell_ids, local_face_ids = np.meshgrid(np.arange(num_cells), range(num_faces_per_cell), indexing="ij")

    # For each cell (defined by vertex IDs), use the FACE_DEFINITIONS lookup to create a list of all faces.
    # Initially, we have shape (num_cells, num_faces, num_vertices). 
    cell_face_vertices = np.asarray(cell_block.data[:, FACE_DEFINITIONS[cell_block.type]], dtype=jnp.int32)

    # Now we flatten everything, for ease of use!
    cell_ids = np.reshape(cell_ids, (-1))
    local_face_ids = np.reshape(local_face_ids, (-1))
    # We flatten this down into (num_cells * num_faces, num_vertices), where we know num_vertices == 3.
    cell_face_vertices = np.reshape(cell_face_vertices, (-1, num_verts_per_face))
    return cell_face_vertices, cell_ids


@app.cell
def _(cell_face_vertices):
    # Sorting the vertices of each face by their ID gives the "hash" of each face.
    cell_face_hashes, cell_face_parity = jax.vmap(sort3_with_parity_bit)(cell_face_vertices)
    return cell_face_hashes, cell_face_parity


@app.cell
def _(cell_face_hashes):
    face_hashes_order = np.lexsort(cell_face_hashes[..., ::-1].T)

    # Sort all our data (so we can keep track of it)
    sorted_face_hashes = cell_face_hashes[face_hashes_order, :]
    return face_hashes_order, sorted_face_hashes


@app.cell
def _(sorted_face_hashes):
    # Assumption -> Faces (or their hashes) appear AT MOST twice.
    unique_face_hashes, sorted_face_ids, face_lexkey_counts = np.unique(
        sorted_face_hashes,
        return_inverse=True, 
        return_counts=True, 
        sorted=True, 
        axis=0
    )
    assert np.max(face_lexkey_counts) <= 2, "Invalid mesh: Faces appear connected to more than 2 cells"
    return face_lexkey_counts, sorted_face_ids


@app.cell
def _(face_hashes_order, sorted_face_ids):
    cell_face_ids = np.empty_like(sorted_face_ids)
    cell_face_ids[face_hashes_order] = sorted_face_ids
    return (cell_face_ids,)


@app.cell
def _(cell_face_parity, face_lexkey_counts):
    num_faces = len(face_lexkey_counts)

    # I don't *think* it's a given that these will always be equal... 
    # An _internal_ face must be always be counted for each.
    # But _external_ faces will only count towards either -1 or +1, not both.
    print("Number of faces that are referenced with parity")
    print(f"\t even: {np.count_nonzero(cell_face_parity == 0):10,}")
    print(f"\t  odd: {np.count_nonzero(cell_face_parity == 1):10,}")
    # Regardless, the number times a face is stored as -1/+1 on cells doesn't equal the number of faces.
    print(f"Total number of faces: {num_faces:10,}")
    # Let's check external vs internal
    print(f"Number of external faces (referenced by cells once):  {np.count_nonzero(face_lexkey_counts == 1):10,}")
    print(f"Number of internal faces (referenced by cells twice): {np.count_nonzero(face_lexkey_counts == 2):10,}")
    return (num_faces,)


@app.cell
def _(cell_face_ids, cell_face_parity, cell_ids, num_faces):
    # We know there can be a maximum of 2 cells per face
    # Let's define the cell with parity=-1 as column 0, and parity=+1 as column 1
    face_cell_ids = np.full((num_faces, 2), fill_value=-1, dtype=int)
    face_cell_ids[cell_face_ids, cell_face_parity] = cell_ids
    return (face_cell_ids,)


@app.cell
def _(cell_face_ids, cell_face_parity, cell_ids, face_cell_ids):
    assert np.all(face_cell_ids[cell_face_ids, cell_face_parity] == cell_ids), "Indexing is messed up"
    cell_to_cell = face_cell_ids[cell_face_ids, 1 - cell_face_parity]
    return (cell_to_cell,)


@app.cell
def _(cell_to_cell, num_faces_per_cell):
    cell_to_cell.reshape((-1, num_faces_per_cell))
    return


if __name__ == "__main__":
    app.run()
