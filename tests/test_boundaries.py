import jax
import jax.numpy as jnp
import numpy as np
from pytest import approx, raises
from test_transport import two_quad_mesh

from raytrax.grid import apply_boundaries, match_faces
from raytrax.gridtypes import Mesh
from raytrax.intersections import LinearRay
from raytrax.sampling import cosine_hemisphere, select_emission
from raytrax.transport import trace


def test_match_faces():
    mesh = two_quad_mesh()
    # Unsorted vertex order must still match
    face_ids = match_faces(mesh.topology.faces, np.array([[1, 0], [4, 1], [3, 0]]))
    assert np.all(np.asarray(mesh.topology.faces.vertices[face_ids]) == [[0, 1], [1, 4], [0, 3]])
    with raises(ValueError):
        match_faces(mesh.topology.faces, np.array([[0, 4]]))  # diagonal, not a face


def test_cosine_hemisphere():
    normal = jnp.array([0.0, 0.0, 1.0])
    keys = jax.random.split(jax.random.key(seed=0), 20_000)
    directions = jax.vmap(cosine_hemisphere, in_axes=(0, None))(keys, normal)
    cosines = directions @ normal
    assert np.all(cosines > 0.0)
    assert np.mean(cosines) == approx(2 / 3, abs=0.01)


def test_select_emission_from_boundary():
    mesh = two_quad_mesh()
    # Hot bottom edges of both cells, cold gas
    face_ids = jnp.asarray(match_faces(mesh.topology.faces, np.array([[0, 1], [1, 2]])))
    cell_field = jnp.zeros(2)
    face_field = jnp.ones(2)

    keys = jax.random.split(jax.random.key(seed=1), 200)
    rays, cell_ids = jax.vmap(select_emission, in_axes=(0, None, None, None, None))(
        keys, cell_field, face_field, face_ids, mesh
    )
    assert np.all(rays.terminus[:, 1] == approx(0.0))  # On the bottom boundary
    assert np.all((rays.terminus[:, 0] >= 0.0) & (rays.terminus[:, 0] <= 2.0))
    assert np.all(rays.tangent[:, 1] > 0.0)  # Inward
    assert np.all(cell_ids == np.floor(rays.terminus[:, 0]).clip(max=1))


def test_apply_boundaries_sentinels():
    mesh = two_quad_mesh()
    topology = jax.tree.map(np.asarray, mesh.topology)
    bottom = match_faces(topology.faces, np.array([[0, 1], [1, 2]]))
    right = match_faces(topology.faces, np.array([[2, 5]]))
    topology, sentinels = apply_boundaries(topology, {"hot": bottom, "wall": right})
    assert sentinels == {"hot": -2, "wall": -3}
    neighbours = np.asarray(topology.cells.neighbours)
    assert np.count_nonzero(neighbours == -2) == 2
    assert np.count_nonzero(neighbours == -3) == 1
    assert np.count_nonzero(neighbours == -1) == 3  # Untagged boundaries remain
    with raises(ValueError):
        internal = match_faces(topology.faces, np.array([[1, 4]]))
        apply_boundaries(topology, {"bad": internal})


def test_trace_accumulates_on_wall():
    mesh = two_quad_mesh()
    topology = jax.tree.map(np.asarray, mesh.topology)
    right = match_faces(topology.faces, np.array([[2, 5]]))
    topology, sentinels = apply_boundaries(topology, {"wall": right})
    mesh = jax.tree.map(jnp.asarray, Mesh(geometry=mesh.geometry, topology=topology))

    rays = LinearRay(
        terminus=jnp.array([[0.5, 0.5]]),
        tangent=jnp.array([[1.0, 0.0]]),
        travel=jnp.zeros(1),
    )
    cell_ids, rays, remaining, cell_deposits, face_deposits = trace(
        jnp.array([0]),
        rays,
        jnp.ones(1),
        mesh,
        optical_thickness=1.0,
        wall_sentinel=sentinels["wall"],
        num_steps=10,
    )
    assert cell_ids[0] == sentinels["wall"]
    assert remaining[0] == approx(0.0)
    assert face_deposits[right[0]] == approx(np.exp(-1.5))  # 0.5 + 1.0 travelled
    assert cell_deposits.sum() + face_deposits.sum() == approx(1.0)
