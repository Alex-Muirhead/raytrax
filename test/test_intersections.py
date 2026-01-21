import jax
import jax.numpy as jnp
import numpy as np
from equinox._misc import currently_jitting
from pytest import approx, mark, param

from intersections import ConvexCell, LinearRay, crossing


def unit_vec(a: list[float] | np.NDArray):
    vec = np.asarray(a)
    return vec / np.linalg.vector_norm(vec)


def unit_2D_cell(x: int = 0, y: int = 0) -> ConvexCell:
    """Generate a unit-square cell.

    Square axis-aligned cell with unit side lengths.
    Can be offset.
        1
      +---+
    2 |   | 0
      +---+
        3

    Params:
        x (int): The lower-left x-coordinate
        y (int): The lower-left y-coordinate
    """
    # fmt: off
    backend = jnp if currently_jitting() else np 
    normal = backend.array(
        [[+1.0,  0.0],
         [ 0.0, +1.0],
         [-1.0,  0.0],
         [ 0.0, -1.0]],
        dtype=float
    )
    offset = backend.array(
        [x + 1, y + 1, x, y],
        dtype=float
    )
    # fmt: on
    return ConvexCell(normal=normal, offset=offset)


def unit_3D_cell(x: int = 0, y: int = 0, z: int = 0) -> ConvexCell:
    """Generate a unit-cube cell."""
    # fmt: off
    backend = jnp if currently_jitting() else np 
    normal = backend.array(
        [[+1.0,  0.0,  0.0],
         [ 0.0, +1.0,  0.0],
         [ 0.0,  0.0, +1.0],
         [-1.0,  0.0,  0.0],
         [ 0.0, -1.0,  0.0],
         [ 0.0,  0.0, -1.0]],
        dtype=float
    )
    offset = backend.array(
        [x + 1, y + 1, z + 1, x, y, z],
        dtype=float
    )
    # fmt: on
    return ConvexCell(normal=normal, offset=offset)


def unit_2D_grid(nx: int, ny: int) -> ConvexCell:
    """Generate a grid of unit-square cells."""
    coords = np.meshgrid(range(nx), range(ny), indexing="ij")
    coords = map(np.ravel, coords)
    return jax.vmap(unit_2D_cell)(*coords)


def unit_3D_grid(nx: int, ny: int, nz: int) -> ConvexCell:
    """Generate a grid of unit-square cells."""
    coords = np.meshgrid(range(nx), range(ny), range(nz), indexing="ij")
    coords = map(np.ravel, coords)
    return jax.vmap(unit_3D_cell)(*coords)


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
    normal = ...
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


testdata = [
    param(np.array([+1.0, 0.0]), 0, 0.5, id="E"),
    param(np.array([0.0, +1.0]), 1, 0.5, id="N"),
    param(np.array([-1.0, 0.0]), 2, 0.5, id="W"),
    param(np.array([0.0, -1.0]), 3, 0.5, id="S"),
    # It will *always* be the small index of the faces (i.e. S or E => E)
    param(unit_vec([+1.0, +1.0]), 0, np.sqrt(1 / 2), id="NE"),
    param(unit_vec([-1.0, +1.0]), 1, np.sqrt(1 / 2), id="NW"),
    param(unit_vec([-1.0, -1.0]), 2, np.sqrt(1 / 2), id="SW"),
    param(unit_vec([+1.0, -1.0]), 0, np.sqrt(1 / 2), id="SE"),
]


@mark.parametrize("direction,expected_face,expected_travel", testdata)
def test_2D_crossing_from_center(direction, expected_face, expected_travel):
    ray = LinearRay(
        terminus=np.array([0.5, 0.5]),
        tangent=direction,
        travel=0.0,  # Does this need to be an array too?
    )
    cell = unit_2D_cell()

    actual_face, actual_travel = crossing(cell, ray)
    assert actual_face == expected_face
    assert actual_travel == approx(expected_travel)


def test_2D_crossing_poor_alignment():
    # Start inside the cell, on the facet
    # We point *just* across the facet
    ray = LinearRay(
        terminus=np.array([0.0, 0.5]),  # West facet, index=2
        tangent=unit_vec([-1e-7, 1.0]),
        travel=0.0,
    )
    # Our alignment is small (1e-10 for testing)
    cell = unit_2D_cell()

    # We expect to skip the problematic west boundary
    face, travel = crossing(cell, ray, epsilon=1e-6)
    assert face == 1
    assert travel == approx(0.5)
