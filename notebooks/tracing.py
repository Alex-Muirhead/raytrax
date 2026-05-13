import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    from copy import replace

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import lox
    import marimo as mo
    import matplotlib.pyplot as plt
    import meshio
    import numpy as np

    from raytrax.intersections import LinearRay, ConvexCell, crossing
    from raytrax.grid import process_cell_block
    from raytrax.gridtypes import Mesh


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
        import contextlib, io

        # Noisy meshio.read
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            input_mesh = meshio.read("../plane.msh")
        if stdout := f.getvalue().strip():
            print(stdout)
        
        # Trim the z-coordinate
        vertices = input_mesh.points[..., :2]
        cell_block = input_mesh.cells[0]
    
        ndim = 2
        num_cells = len(cell_block)
    return cell_block, ndim, num_cells, vertices


@app.cell
def _(cell_block, vertices):
    mesh_topo = process_cell_block(cell_block)
    mesh_geom = mesh_topo.build_geometric(vertex_coords=vertices)
    mesh = Mesh(geometry=mesh_geom, topology=mesh_topo)
    mesh = jax.tree.map(jnp.asarray, mesh)
    return mesh, mesh_geom, mesh_topo


@app.cell
def _(mesh):
    face_geom = mesh.faces.geometry
    cell_topo = mesh.cells.topology

    convex_cells = ConvexCell(
        normal=face_geom.normal[cell_topo.face_ids] * cell_topo.face_signs[..., None],
        offset=face_geom.offset[cell_topo.face_ids] * cell_topo.face_signs,
    )
    return cell_topo, convex_cells


@app.cell
def _(num_cells):
    new_cell_datacell_data = jnp.zeros(num_cells)
    return


@app.cell
def _(convex_cells):
    def walking(cell_id, ray: LinearRay, mesh: Mesh):
        cell = convex_cells[cell_id]
        out_face, distance = crossing(cell=cell, ray=ray)
        next_cell_id = mesh.cells[cell_id].topology.neighbours[out_face]
        next_ray = replace(ray, travel=ray.travel+distance)
        return next_cell_id, next_ray, distance

    return (walking,)


@app.cell
def _(mesh_geom, ndim, num_cells):
    key = jax.random.key(seed=4)
    key, key_cells, key_rays = jax.random.split(key, 3)
    num_traces = 5000

    cell_ids = jax.random.choice(key_cells, num_cells, shape=(num_traces,))

    key_terminus, key_tangent = jax.random.split(key_rays, ndim)
    ray_terminus = jnp.asarray(mesh_geom.cells.centroid[cell_ids])
    ray_tangents = normalise(jax.random.normal(key_tangent, shape=(num_traces, ndim)))
    ray_travel = jnp.zeros((num_traces,))
    rays = LinearRay(terminus=ray_terminus, tangent=ray_tangents, travel=ray_travel)
    return cell_ids, rays


@app.cell
def _(cell_data, cell_ids, mesh, rays, walking):
    new_cell_ids = cell_ids
    old_cell_ids = cell_ids
    new_rays = rays
    new_cell_data = cell_data

    for _ in range(20):
        mid_cell_ids, new_rays, distance = jax.vmap(lox.tap(walking), in_axes=(0, 0, None))(old_cell_ids, new_rays, mesh)
        new_cell_data = new_cell_data.at[old_cell_ids].add(distance)
        new_cell_ids = jnp.where(mid_cell_ids != -1, mid_cell_ids, old_cell_ids)
        old_cell_ids = new_cell_ids

    print(jnp.count_nonzero(mid_cell_ids != -1))
    print(distance)
    return new_cell_data, new_rays


@app.cell
def _(mesh_topo, new_rays, rays, vertices):
    from matplotlib.collections import LineCollection

    _fig, _ax = plt.subplots()

    edges = LineCollection(vertices[mesh_topo.faces.vertices], linewidths=0.5, colors='k')
    paths = LineCollection(
        np.stack([rays.p, new_rays.p], axis=1), 
        cmap="viridis"
    )
    paths.set_array(new_rays.travel)

    _ax.add_collection(paths)
    _ax.add_collection(edges)
    _ax.set_aspect("equal")
    _ax.autoscale_view()
    _ax
    return


@app.cell
def _(cell_topo, new_cell_data, vertices):
    from matplotlib.collections import PolyCollection

    _fig, _ax = plt.subplots()

    polygons = PolyCollection(vertices[cell_topo.vertices], cmap="viridis")
    polygons.set_array(new_cell_data)
    _ax.add_collection(polygons)
    _ax.autoscale_view()
    _fig.colorbar(polygons, ax=_ax)
    _ax
    return


if __name__ == "__main__":
    app.run()
