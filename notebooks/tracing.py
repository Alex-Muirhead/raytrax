import marimo

__generated_with = "0.23.10"
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
    from matplotlib.collections import LineCollection, PolyCollection

    from raytrax.intersections import LinearRay, ConvexCell, crossing
    from raytrax.grid import process_cell_block
    from raytrax.gridtypes import Mesh, Cell, Face
    from raytrax.random import sphere, simplex


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
    face_geom = mesh.faces.geometry
    cell_topo = mesh.cells.topology

    convex_cells = ConvexCell(
        normal=face_geom.normal[cell_topo.face_ids] * cell_topo.face_signs[..., None],
        offset=face_geom.offset[cell_topo.face_ids] * cell_topo.face_signs,
    )
    return cell_topo, convex_cells


@app.cell
def _(num_cells):
    cell_data = jnp.zeros(num_cells)
    return


@app.cell
def _(convex_cells):
    def walking(cell_id: int, ray: LinearRay, mesh: Mesh):
        cell = convex_cells[cell_id]
        out_face, distance = crossing(cell=cell, ray=ray)
        next_cell_id = mesh.cells[cell_id].topology.neighbours[out_face]
        next_ray = replace(ray, travel=ray.travel+distance)
        return next_cell_id, next_ray, distance

    return (walking,)


@app.cell
def _(ndim):
    def select_point(key, cell: Cell, faces: Face, vertices):
        subcell_key, barycentric_key = jax.random.split(key, num=2)
        face_id = jax.random.choice(key=subcell_key, a=cell.topology.face_ids, p=cell.geometry.face_weights)
        vertex_points = vertices[faces[face_id].topology.vertices]
        vertex_weights = simplex(key=barycentric_key, ndim=ndim)
        return vertex_weights[0] * cell.geometry.centroid + vertex_weights[1:] @ vertex_points

    return (select_point,)


@app.cell
def _(ndim, num_cells, select_point):
    def select_start(key, mesh: Mesh):
        key_cell, key_point, key_direction = jax.random.split(key, 3)
        cell_id = jax.random.choice(key_cell, num_cells, p=mesh.cells.geometry.volume)
        terminus = select_point(key_point, mesh.cells[cell_id], mesh.faces, mesh.verts)
        tangent = sphere(key_direction, ndim)
        ray = LinearRay(terminus=terminus, tangent=tangent, travel=jnp.zeros(()))
        return ray, cell_id

    return (select_start,)


@app.cell
def _(Self):
    class RayState(eqx.Module):
        energy: jax.Array
        cell_id: jax.Array

        @classmethod
        def new(cls, cell_id: jax.Array) -> Self:
            shape = cell_id.shape
            return cls(energy=jnp.zeros(shape), cell_id=cell_id)

    return


@app.cell
def _(mesh, num_cells, select_start, walking):
    _key = jax.random.key(seed=4)
    key, key_cells, key_rays = jax.random.split(_key, 3)
    num_traces = 500

    keys = jax.random.split(key, num_traces)
    rays, cell_ids = jax.vmap(jax.jit(select_start), in_axes=(0, None))(keys, mesh)

    optical_thickess = 1.0
    cell_energies = jnp.zeros(shape=(num_cells,))
    ray_energies = jnp.where(rays.terminus[:, 0] < 0, 1.0, 0.0)

    def step(cell_id, ray, ray_energy, mesh):
        new_cell_id, new_ray, distance = walking(cell_id, ray, mesh)
        distance = jnp.where(cell_id == -1, 0.0, distance)
        cell_id = jnp.where(cell_id == -1, -1, new_cell_id)
        ray = replace(ray, travel=jnp.where(cell_id == -1, ray.travel, new_ray.travel))
        optical_distance = optical_thickess * distance
        ray_energy = ray_energy * jnp.exp(-optical_distance)
        energy_decay = ray_energy * (1 - jnp.exp(-optical_distance))
        return new_cell_id, ray, ray_energy, energy_decay

    def collect_step(cell_ids, rays, ray_energies, cell_energies, mesh):
        new_cell_ids, rays, ray_energies, energy_dropped = jax.vmap(jax.jit(step), in_axes=(0, 0, 0, None))(cell_ids, rays, ray_energies, mesh)
        # This *must* be *outside* of the vmap, otherwise all HELL breaks loose with allocs!
        cell_energies = cell_energies.at[cell_ids].add(energy_dropped)
        return new_cell_ids, rays, ray_energies, cell_energies, mesh

    def wrapped_collect_step(i, state):
        return collect_step(*state)

    init_state = (cell_ids, rays, ray_energies, cell_energies, mesh)
    final_state = jax.lax.fori_loop(0, 6, wrapped_collect_step, init_state)
    new_cell_ids, new_rays, new_ray_energies, new_cell_energies, _ = final_state
    return new_cell_energies, new_cell_ids, new_ray_energies, new_rays, rays


@app.cell
def _(new_ray_energies):
    jnp.where(jnp.isnan(new_ray_energies))
    return


@app.cell
def _(new_cell_ids):
    new_cell_ids[167]
    return


@app.cell
def _(new_rays):
    new_rays.travel[167]
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

    # _()
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
    input_mesh.cell_data['energy'] = np.asarray((new_cell_energies / mesh.cells.geometry.volume))
    input_mesh.write("../cylinder.vtk")
    return


if __name__ == "__main__":
    app.run()
