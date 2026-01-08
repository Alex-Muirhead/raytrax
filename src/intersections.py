import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


class ConvexCell(eqx.Module):
    normals: Float[Array, "nfaces ndim"]
    offsets: Float[Array, " nfaces"]

    @jax.jit(static_argnames="epsilon")
    def contains(self, points: Float[Array, "npoints ndim"], epsilon: float = 0.0):
        if epsilon == 0.0:
            epsilon = np.finfo(points.dtype).resolution
        point_offsets = jnp.dot(points, self.normals.T)  # [npoints nfaces]
        return jnp.all(point_offsets <= self.offsets + epsilon, axis=-1)


# ================================ UNITTESTS ================================


def unit_2D_cell() -> ConvexCell:
    """Generate a unit-square cell."""
    # fmt: off
    normals = np.array(
        [[+1.0,  0.0],
         [ 0.0, +1.0],
         [-1.0,  0.0],
         [ 0.0, -1.0]],
        dtype=float
    )
    offsets = np.array(
        [1.0, 1.0, 0.0, 0.0],
        dtype=float
    )
    # fmt: on
    return ConvexCell(normals=normals, offsets=offsets)


def unit_3D_cell() -> ConvexCell:
    """Generate a unit-cube cell."""
    # fmt: off
    normals = np.array(
        [[+1.0,  0.0,  0.0],
         [ 0.0, +1.0,  0.0],
         [ 0.0,  0.0, +1.0],
         [-1.0,  0.0,  0.0],
         [ 0.0, -1.0,  0.0],
         [ 0.0,  0.0, -1.0]],
        dtype=float
    )
    offsets = np.array(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        dtype=float
    )
    # fmt: on
    return ConvexCell(normals=normals, offsets=offsets)


def random_2D_quad() -> ConvexCell:
    # fmt: off
    vertices = np.array(
        [[0.0, 0.0],
         [1.0, 0.0],
         [1.0, 1.0],
         [0.0, 1.0]],
        dtype=float
    )
    interfaces = np.array(
        [[0, 1],
         [1, 2],
         [2, 3],
         [3, 4]],
        dtype=int
    )
    # fmt: on
    normals = ...
    raise NotImplementedError()


def print_log(log: dict[str, object]) -> None:
    for key, value in log.items():
        print(key + ":")
        print(value, value.shape)
        print("")


def test_2D_containment_simple():
    # fmt: off
    points = np.array(
        [[0.5 , 0.5 ],
         [0.25, 0.9 ],
         [0.1 , 0.75],
         [ 1.5, 1.5 ]],
        dtype=float
    )
    # fmt: on
    cell = unit_2D_cell()
    actual = cell.contains(points)
    expected = np.array([True, True, True, False])

    assert np.all(actual == expected)


def test_3D_containment_simple():
    # fmt: off
    points = np.array(
        [[0.5 , 0.5 , 0.5 ],
         [0.25, 0.9 , 0.1 ],
         [0.1 , 0.75, 0.75],
         [ 1.5, 1.5 , 1.5 ]],
        dtype=float
    )
    # fmt: on
    cell = unit_3D_cell()
    actual = cell.contains(points)
    expected = np.array([True, True, True, False])

    assert np.all(actual == expected)


def test_2D_containment_edges():
    # fmt: off
    points = np.array(
        [[ 1.0 ,  1.0 ],
         [ 0.0 ,  0.0 ],
         [ 0.5 ,  1.0 ],
         [-1E-6, -1E-6]],
        dtype=float
    )
    # fmt: on
    cell = unit_2D_cell()

    actual = cell.contains(points, epsilon=0.001)
    expected = np.array([True, True, True, True])

    assert np.all(actual == expected)


def test_3D_containment_edges():
    # fmt: off
    points = np.array(
        [[ 1.0 ,  1.0 ,  1.0 ],
         [ 0.0 ,  0.0 ,  0.0 ],
         [ 0.5 ,  1.0 ,  0.5 ],
         [-1E-6, -1E-6, -1E-6]],
        dtype=float
    )
    # fmt: on
    cell = unit_3D_cell()

    actual = cell.contains(points, epsilon=0.001)
    expected = np.array([True, True, True, True])

    assert np.all(actual == expected)
