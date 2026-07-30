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
    from scipy.special import expn

    from raytrax.grid import apply_boundaries, boundary_groups, match_faces, process_cell_block
    from raytrax.gridtypes import Mesh
    from raytrax.sampling import select_emission
    from raytrax.transport import Boundary, boundary_kinds, trace


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

        _y, _x = np.mgrid[0 : height + 1, 0 : width + 1]
        vertices = np.column_stack([_x.flatten(), _y.flatten()]).astype(float)
        vertex_ids = np.arange(_x.size).reshape(_x.shape)

        quad = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 1],
                [1, 0],
            ]
        )

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

        mesh_file = mo.watch.file("../cylinder.msh")

        # Noisy meshio.read
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            input_mesh = meshio.read(mesh_file)
        if stdout := f.getvalue().strip():
            print(stdout)

        ndim = 3
        # Trim the z-coordinate
        vertices = input_mesh.points[..., :ndim]
        cell_block = next(block for block in input_mesh.cells if block.type == "tetra")

        num_cells = len(cell_block)
    return cell_block, input_mesh, num_cells, vertices


@app.cell
def _(input_mesh):
    input_mesh.points
    return


@app.cell
def _(cell_block, input_mesh, vertices):
    mesh_topo = process_cell_block(cell_block)
    boundaries = boundary_groups(mesh_topo, input_mesh)
    mesh_topo, sentinels = apply_boundaries(mesh_topo, boundaries)
    mesh_geom = mesh_topo.build_geometric(vertex_coords=vertices)
    mesh = Mesh(geometry=mesh_geom, topology=mesh_topo)
    mesh = jax.tree.map(jnp.asarray, mesh)
    return boundaries, mesh, mesh_geom, mesh_topo, sentinels


@app.cell
def _(mesh):
    cell_topo = mesh.cells.topology
    return (cell_topo,)


@app.cell
def _(boundaries):
    hot_face_ids = jnp.asarray(boundaries["hot_end"])
    return (hot_face_ids,)


@app.cell
def _(sentinels):
    # Behaviour of each tagged boundary group. Switch the lateral surface to
    # Boundary.SYMMETRY to mirror rays back in, which recovers the 1D tangent-slab
    # profile at every radius (needs num_steps >= 200 to converge).
    lateral_kind = Boundary.ESCAPE
    kinds = boundary_kinds(
        sentinels,
        {"hot_end": Boundary.WALL, "wall": Boundary.WALL, "lateral": lateral_kind},
    )
    return (kinds,)


@app.cell
def _(hot_face_ids, kinds, mesh, num_cells):
    _key = jax.random.key(seed=4)
    key, key_cells, key_rays = jax.random.split(_key, 3)
    num_traces = 5_000_000

    # Cold gas; hot end cap with sigma * T^4 = 1
    cell_energies = jnp.zeros(num_cells)
    face_energies = mesh.geometry.faces.area[hot_face_ids]

    keys = jax.random.split(key, num_traces)
    rays, cell_ids = jax.vmap(jax.jit(select_emission), in_axes=(0, None, None, None, None))(
        keys, cell_energies, face_energies, hot_face_ids, mesh
    )

    total_energy = cell_energies.sum() + face_energies.sum()
    ray_energies = jnp.full(num_traces, fill_value=total_energy / num_traces)
    optical_thickness = 1.0

    new_cell_ids, new_rays, new_ray_energies, new_cell_energies, new_face_energies = trace(
        cell_ids, rays, ray_energies, mesh, optical_thickness, kinds, num_steps=100
    )
    return (
        new_cell_energies,
        new_cell_ids,
        new_face_energies,
        new_ray_energies,
        new_rays,
        optical_thickness,
        ray_energies,
        rays,
    )


@app.cell
def _(mesh, new_cell_energies, optical_thickness, vertices):
    def _():
        fig, ax = plt.subplots()

        x = mesh.geometry.cells.centroid[:, 0]
        heating = new_cell_energies / mesh.geometry.cells.volume
        ax.scatter(x, heating, s=1, alpha=0.2, label="Monte-Carlo")

        # Tangent slab: black wall at the lower x bound, absorption only
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        depth = np.linspace(0.0, x_max - x_min, 200)
        slab = 2 * optical_thickness * expn(2, optical_thickness * depth)
        ax.plot(x_min + depth, slab, c="C1", label="Tangent slab $2\\kappa E_2(\\kappa x)$")

        ax.set_xlabel("x")
        ax.set_ylabel("Volumetric heating")
        ax.legend()
        return ax

    # This plot takes too long to run, there are too many points. We don't need to plot them all.
    # _()
    return


@app.cell
def _(mesh, new_cell_energies, optical_thickness, vertices):
    def _():
        fig, ax = plt.subplots()

        # Volume-weighted mean heating in slabs spanning the axial extent
        x = mesh.geometry.cells.centroid[:, 0]
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        bins = jnp.linspace(x_min, x_max, 21)
        idx = jnp.digitize(x, bins) - 1
        slab_energy = jnp.zeros(bins.size - 1).at[idx].add(new_cell_energies)
        slab_volume = jnp.zeros(bins.size - 1).at[idx].add(mesh.geometry.cells.volume)
        ax.stairs(slab_energy / slab_volume, bins, label="Monte-Carlo (binned)")

        depth = np.linspace(0.0, x_max - x_min, 200)
        slab = 2 * optical_thickness * expn(2, optical_thickness * depth)
        ax.plot(x_min + depth, slab, c="C1", label="Tangent slab $2\\kappa E_2(\\kappa x)$")

        ax.set_xlabel("x")
        ax.set_ylabel("Volumetric heating")
        ax.legend()
        return ax


    _()
    return


@app.cell
def _(mesh, new_cell_energies, optical_thickness, vertices):
    def _():
        fig, ax = plt.subplots()

        # Volume-weighted mean heating in radial shells, split by axial slab.
        # Bin edges come from the vertex bounds, so they follow the mesh dimensions.
        x = mesh.geometry.cells.centroid[:, 0]
        radius = jnp.linalg.vector_norm(mesh.geometry.cells.centroid[:, 1:], axis=-1)

        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        r_max = np.linalg.norm(vertices[:, 1:], axis=-1).max()
        x_bins = np.linspace(x_min, x_max, 5)
        r_bins = jnp.linspace(0.0, r_max, 51)
        idx = (jnp.digitize(x, x_bins) - 1, jnp.digitize(radius, r_bins) - 1)

        shape = (x_bins.size - 1, r_bins.size - 1)
        shell_energy = jnp.zeros(shape).at[idx].add(new_cell_energies)
        shell_volume = jnp.zeros(shape).at[idx].add(mesh.geometry.cells.volume)

        for _i, (_lo, _hi) in enumerate(zip(x_bins[:-1], x_bins[1:])):
            ax.stairs(shell_energy[_i] / shell_volume[_i], r_bins, color=f"C{_i}", label=f"${_lo:+.2f} < x < {_hi:+.2f}$")
            # The tangent slab is radially uniform, so compare against its slab mean.
            # Integrating 2 kappa E_2 exactly gives 2 (E_3(a) - E_3(b)) / (b - a).
            _depth = optical_thickness * (np.array([_lo, _hi]) - x_min)
            _slab = 2 * (expn(3, _depth[0]) - expn(3, _depth[1])) / (_hi - _lo)
            ax.axhline(_slab, color=f"C{_i}", ls="--", lw=0.8)

        ax.set_xlabel("r")
        ax.set_ylabel("Volumetric heating")
        ax.legend()
        return ax


    _()
    return


@app.cell
def _(
    new_cell_energies,
    new_cell_ids,
    new_face_energies,
    new_ray_energies,
    new_rays,
    ray_energies,
):
    # Conservation: gas + wall + escaped + still in flight.
    # A non-negative final cell id means the ray ran out of steps inside the mesh,
    # which symmetry boundaries make far more likely.
    in_flight = new_cell_ids >= 0
    print(f"input:     {ray_energies.sum():.6f}")
    print(f"gas:       {new_cell_energies.sum():.6f}")
    print(f"wall:      {new_face_energies.sum():.6f}")
    print(f"escaped:   {jnp.where(in_flight, 0.0, new_ray_energies).sum():.6f}")
    print(f"in flight: {jnp.where(in_flight, new_ray_energies, 0.0).sum():.6f}")
    print(f"rays in flight: {in_flight.mean():.2%}, max travel: {new_rays.travel.max():.3f}")
    return


@app.cell
def _(mesh_topo, new_rays, rays, vertices):
    def _():
        fig, ax = plt.subplots()

        # Project onto the xy-plane. With symmetry boundaries the ray reflects, so
        # this chord is start-to-finish rather than the path actually flown.
        edges = LineCollection(vertices[mesh_topo.faces.vertices][..., :2], linewidths=0.5, colors="k")
        paths = LineCollection(np.stack([rays.terminus, new_rays.terminus], axis=1)[..., :2], cmap="viridis")
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

        edges = LineCollection(vertices[mesh_topo.faces.vertices], linewidths=0.5, colors="k")
        ax.add_collection(edges)

        ax.scatter(*ray_terminus.T, s=0.1)
        ax.scatter(*mesh_geom.cells.centroid.T, s=2)
        ax.set_aspect("equal")
        ax.autoscale_view()
        return ax

    # _()
    return


@app.cell
def _(input_mesh, mesh, mesh_topo, new_cell_energies, new_face_energies):
    # One heating array per block: volumetric for cells, per-area for faces
    _heating = []
    for _block in input_mesh.cells:
        if _block.type == "tetra":
            _heating.append(np.asarray(new_cell_energies / mesh.geometry.cells.volume))
        else:
            _ids = match_faces(mesh_topo.faces, _block.data)
            _heating.append(np.asarray(new_face_energies[_ids] / mesh.geometry.faces.area[_ids]))

    output_mesh = meshio.Mesh(
        points=input_mesh.points,
        cells=input_mesh.cells,
        cell_data={"heating": _heating},
    )
    output_mesh.write("../cylinder.vtk")
    return


if __name__ == "__main__":
    app.run()
