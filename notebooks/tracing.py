import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import marimo as mo
    import meshio
    import numpy as np

    from raytrax.grid import process_cell_block

    return jax, meshio, mo, np, process_cell_block


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We generate a simple rectangular grid of vertices
    """)
    return


@app.cell
def _(meshio, np):
    width, height = 3, 3

    _x, _y = np.mgrid[0:width+1, 0:height+1]
    vertices = np.column_stack([_x.flatten(), _y.flatten()]).astype(float)
    vertex_ids = np.arange(_x.size).reshape(_x.shape)

    quad = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
    ])

    cells = []

    for _i in range(height):
        for _j in range(width):
            cell = vertex_ids[np.unstack(quad + np.array([_i, _j]), axis=-1)]
            cells.append(cell)

    cells = np.array(cells)
    cell_block = meshio.CellBlock("quad", cells)
    return cell_block, vertices


@app.cell
def _(cell_block, process_cell_block, vertices):
    mesh_topo = process_cell_block(cell_block)
    mesh_geom = mesh_topo.build_geometric(vertex_coords=vertices)
    return mesh_geom, mesh_topo


@app.cell
def _(jax, mesh_geom, mesh_topo):
    cell_ids = [1, 2, 3]

    cell_topo = jax.tree.map(lambda leaf: leaf[cell_ids], mesh_topo.cells)
    cell_geom = jax.tree.map(lambda leaf: leaf[cell_ids], mesh_geom.cells)

    cell_faces = jax.tree.map(lambda leaf: leaf[cell_topo.face_ids], mesh_geom.faces)
    cell_faces.normal.shape
    return (cell_geom,)


@app.cell
def _(cell_geom, jnp):
    ray_terminus = cell_geom.centroid
    ray_tangent = jnp.array([])
    return


if __name__ == "__main__":
    app.run()
