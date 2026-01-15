import numpy as np
import pytest

from intersections import ConvexCell, LinearRay, crossing


def unit_vec(a: list[float] | np.NDArray):
    vec = np.asarray(a)
    return vec / np.linalg.norm(vec)


def unit_2D_cell(x: int = 0, y: int = 0) -> ConvexCell:
    """Generate a unit-square cell.

    Square axis-alignend cell with unit side lengths.
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
    normal = np.array(
        [[+1.0,  0.0],
         [ 0.0, +1.0],
         [-1.0,  0.0],
         [ 0.0, -1.0]],
        dtype=float
    )
    offset = np.array(
        [x + 1, y + 1, x, y],
        dtype=float
    )
    # fmt: on
    return ConvexCell(normal=normal, offset=offset)


def unit_3D_cell() -> ConvexCell:
    """Generate a unit-cube cell."""
    # fmt: off
    normal = np.array(
        [[+1.0,  0.0,  0.0],
         [ 0.0, +1.0,  0.0],
         [ 0.0,  0.0, +1.0],
         [-1.0,  0.0,  0.0],
         [ 0.0, -1.0,  0.0],
         [ 0.0,  0.0, -1.0]],
        dtype=float
    )
    offset = np.array(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        dtype=float
    )
    # fmt: on
    return ConvexCell(normal=normal, offset=offset)


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
    (np.array([+1.0, 0.0]), 0, 0.5),
    (np.array([0.0, +1.0]), 1, 0.5),
    (np.array([-1.0, 0.0]), 2, 0.5),
    (np.array([0.0, -1.0]), 3, 0.5),
]


@pytest.mark.parametrize("direction,expected_face,expected_travel", testdata)
def test_2D_crossing_from_center(direction, expected_face, expected_travel):
    ray = LinearRay(
        terminus=np.array([0.5, 0.5]),
        tangent=direction,
        travel=0.0,  # Does this need to be an array too?
    )
    cell = unit_2D_cell()

    actual_face, actual_travel = crossing(cell, ray)
    assert actual_face == expected_face
    assert actual_travel == pytest.approx(expected_travel)


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
    assert travel == pytest.approx(0.5)


def test_2D_crossing_poor_precondition():
    # A follow-up to the previous test
    # We crossed the north facet (1)
    ray = LinearRay(
        terminus=np.array([0.0, 0.5]),  # West facet, index=2
        tangent=unit_vec([-1e-7, 1.0]),
        travel=0.5,
    )
    # We now sit left of the south facet (3) of this next cell
    cell = unit_2D_cell(x=0, y=1)

    # Normally, this would take several dozen cells to correct,
    # however we artificially control the epsilon to see the desired
    # result, that the path is correct by crossing the west boundary.
    face, travel = crossing(cell, ray, epsilon=1e-12)
    assert face == 2
    # Hmm... this travel assumes that the boundaries are a regular grid
    assert travel == 0.0
