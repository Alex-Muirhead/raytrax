import marimo

__generated_with = "0.23.10"
app = marimo.App()


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import timeit

    from matplotlib.patches import Polygon, Circle
    from scipy.interpolate import LinearNDInterpolator

    return jax, jnp, plt, timeit


@app.cell
def _():
    from raytrax.random import sphere, simplex

    return simplex, sphere


@app.cell
def _(jax, sphere):
    key = jax.random.key(seed=4)
    num_samples = 10_000
    sphere_samples = sphere(key=key, ndim=3, shape=num_samples)
    return key, num_samples, sphere_samples


@app.cell
def _(jnp, sphere_samples):
    def angular_distance(i, vector):
        alignment = (vector[None, :] * sphere_samples).sum(axis=-1)
        alignment = alignment.at[i].set(jnp.nan)
        best = jnp.arccos(jnp.minimum(jnp.nanmax(alignment), 1.0))
        return i+1, best

    return (angular_distance,)


@app.cell
def _(angular_distance, jax, jnp, num_samples, sphere_samples):
    _, sphere_distances = jax.lax.scan(angular_distance, 0, sphere_samples)
    print(f"Average distance is {jnp.mean(sphere_distances):.3f} rad")
    print(f"Expected distance is {2/jnp.sqrt(num_samples):.3f} rad")
    return (sphere_distances,)


@app.cell
def _(jnp, plt, sphere_distances, sphere_samples):
    flat_samples = sphere_samples[:, :2].copy()
    azimuth_samples = jnp.atan2(sphere_samples[:, 0], sphere_samples[:, 1])
    elevation_samples = jnp.arccos(sphere_samples[:, 2])

    def _():
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        h = ax.scatter(*sphere_samples.T, c=sphere_distances)
        ax.set_aspect("equal")
        fig.colorbar(h)
        return fig

    _()
    return


@app.cell
def _(timeit):
    _timer = timeit.Timer(
        "sphere(key=key, ndim=3, shape=num_samples).block_until_ready()", 
        setup="sphere(key=key, ndim=3, shape=num_samples)", 
        globals=globals(),
    )
    _ = _timer.autorange(callback=lambda n, t: print(f"Amortised time of {t/n*1e6:>3.0f}us over {n:>4} runs"))
    return


@app.cell
def _(jax, jnp):
    _key = jax.random.key(seed=10)
    vertices = jax.random.uniform(_key, shape=(3, 2))
    # Scale triangle to fit in (0,1) box
    vertices -= jnp.min(vertices, axis=0)
    vertices /= jnp.max(vertices, axis=0)
    simplex_area = 1/2 * jnp.abs(jnp.cross(vertices[0, :] - vertices[1, :], vertices[2, :] - vertices[1, :]))
    print(f"Simplex area: {simplex_area:.2f}")
    return simplex_area, vertices


@app.cell
def _(jnp, key, num_samples, simplex, vertices):
    simplex_samples = simplex(key, 2, num_samples)
    simplex_points = simplex_samples @ vertices
    assert jnp.allclose(jnp.sum(simplex_samples, axis=-1), 1.0)
    return (simplex_points,)


@app.cell
def _(jnp, simplex_points):
    def linear_distance(i, vector):
        dist = jnp.linalg.vector_norm(vector[None, :] - simplex_points, axis=-1)
        dist = dist.at[i].set(jnp.nan)
        best = jnp.maximum(jnp.nanmin(dist), 0.0)
        return i+1, best

    return (linear_distance,)


@app.cell
def _(jax, jnp, linear_distance, num_samples, simplex_area, simplex_points):
    _, simplex_distances = jax.lax.scan(linear_distance, 0, simplex_points)
    print(f"Average distance is {jnp.mean(simplex_distances):.3f}")
    print(f"Expected distance is {jnp.sqrt(simplex_area / num_samples / jnp.pi):.3f}")
    return


app._unparsable_cell(
    r"""
    def _():
        fig, ax = plt.subplots()
        h = ax.scatter(*simplex_points.T, c=simplex_distances)
        ax.add_patch(Polygon(vertices, fill=False))
        fig.colorbar(h)simplex_area
        return fig

    _()
    """,
    name="_"
)


@app.cell
def _(timeit):
    _timer = timeit.Timer(
        "simplex(key=key, ndim=2, shape=num_samples).block_until_ready()", 
        setup="simplex(key=key, ndim=2, shape=num_samples)",
        globals=globals(),
    )
    _ = _timer.autorange(callback=lambda n, t: print(f"Amortised time of {t/n*1e6:>3.0f}us over {n:>4} runs"))
    return


if __name__ == "__main__":
    app.run()
