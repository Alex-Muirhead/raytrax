import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")

with app.setup:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import meshio
    import numpy as np
    from matplotlib.collections import LineCollection, PolyCollection

    from raytrax.grid import process_cell_block
    from raytrax.gridtypes import Mesh
    from raytrax.sampling import select_start
    from raytrax.transport import trace


@app.function
def debug_print(log_dict: dict) -> None:
    for key, val in log_dict.items():
        print(key + "\n" + "-" * len(key))
        eqx.tree_pprint(val, short_arrays=False)


@app.function
def normalise(array: jax.Array, *, axis=-1) -> jax.Array:
    lengths = jnp.linalg.vector_norm(array, axis=axis, keepdims=True)
    return array / lengths


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We generate a simple rectangular grid of vertices
    """)
    return


@app.cell
def _():
    if False:
        # Generate our own
        width, height = 15, 15

        _y, _x = np.mgrid[0:height+1, 0:width+1]
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

        ndim = 2
        num_cells = width * height
    else:
        # Read an existing mesh
        import contextlib
        import io

        # Noisy meshio.read
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            input_mesh = meshio.read("../cylinder.msh")
        if stdout := f.getvalue().strip():
            print(stdout)

        ndim = 3
        # Trim the z-coordinate
        vertices = input_mesh.points[..., :ndim]
        cell_block = input_mesh.cells[0]

        num_cells = len(cell_block)
    return cell_block, input_mesh, ndim, num_cells, vertices


@app.cell
def _(cell_block, vertices):
    mesh_topo = process_cell_block(cell_block)
    mesh_geom = mesh_topo.build_geometric(vertex_coords=vertices)
    mesh = Mesh(geometry=mesh_geom, topology=mesh_topo)
    mesh = jax.tree.map(jnp.asarray, mesh)
    return mesh, mesh_geom, mesh_topo


@app.cell
def _(mesh):
    cell_topo = mesh.cells.topology
    return (cell_topo,)


@app.cell
def _(mesh, num_cells):
    _key = jax.random.key(seed=4)
    key, key_cells, key_rays = jax.random.split(_key, 3)
    num_traces = 500_000

    cell_energies = jnp.where(
        mesh.geometry.cells.centroid[:, 0] < 0,
        mesh.geometry.cells.volume,
        0.0,
    )

    keys = jax.random.split(key, num_traces)
    rays, cell_ids = jax.vmap(jax.jit(select_start), in_axes=(0, None, None))(keys, cell_energies, mesh)

    num_rays_per_cell = jnp.zeros(num_cells).at[cell_ids].add(1.0)
    ray_energies = (cell_energies / num_rays_per_cell)[cell_ids]
    optical_thickness = 1.0

    new_cell_ids, new_rays, new_ray_energies, new_cell_energies = trace(
        cell_ids, rays, ray_energies, mesh, optical_thickness, num_steps=100
    )
    return new_cell_energies, new_ray_energies, new_rays, ray_energies, rays


@app.cell
def _(new_ray_energies, ray_energies):
    new_ray_energies / ray_energies
    return


@app.cell
def _(mesh_topo, new_rays, rays, vertices):
    def _():
        fig, ax = plt.subplots()

        edges = LineCollection(vertices[mesh_topo.faces.vertices], linewidths=0.5, colors='k')
        paths = LineCollection(
            np.stack([rays.p, new_rays.p], axis=1), 
            cmap="viridis"
        )
        paths.set_array(new_rays.travel)

        ax.add_collection(paths)
        ax.add_collection(edges)
        ax.set_aspect("equal")
        ax.autoscale_view()
        return ax

    _()
    return


@app.cell
def _(cell_topo, mesh, new_cell_data, vertices):
    def _():
        fig, ax = plt.subplots()

        polygons = PolyCollection(vertices[cell_topo.vertices][..., :2], cmap="viridis")
        polygons.set_array(new_cell_data / mesh.cells.geometry.volume)
        ax.add_collection(polygons)
        ax.autoscale_view()
        fig.colorbar(polygons, ax=ax)
        return ax


    # _()
    return


@app.cell
def _(mesh_geom, mesh_topo, ray_terminus, vertices):
    def _():
        # subplot_kw={"projection": "3d"}
        fig, ax = plt.subplots()

        edges = LineCollection(vertices[mesh_topo.faces.vertices], linewidths=0.5, colors='k')
        ax.add_collection(edges)

        ax.scatter(*ray_terminus.T, s=0.1)
        ax.scatter(*mesh_geom.cells.centroid.T, s=2)
        ax.set_aspect("equal")
        ax.autoscale_view()
        return ax


    # _()
    return


@app.cell
def _(input_mesh, mesh, new_cell_energies):
    input_mesh.cell_data['energy'] = np.asarray(new_cell_energies / mesh.geometry.cells.volume)
    input_mesh.write("../cylinder.vtk")
    return


if __name__ == "__main__":
    app.run()
