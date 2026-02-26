from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class ShapeMixin:
    dimension: int
    num_facets: int
    num_vertices: int

    facet_indices: list[list[int]]


class Shapes(ShapeMixin, Enum):
    # 2D shapes
    TRIANGLE = 2, 3, 3, [[0, 1], [1, 2], [2, 3]]
    QUADRILATERAL = 2, 4, 4, [[0, 1], [1, 2], [2, 3], [3, 0]]
    # 3D shapes
    HEXAHEDRON = (
        3,
        6,
        8,
        [
            [0, 1, 5, 4],  # South
            [4, 5, 7, 6],  # Front
            [1, 4, 7, 5],  # East
            [2, 6, 7, 3],  # North
            [0, 2, 3, 1],  # Back
            [0, 4, 6, 2],  # West
        ],
    )


if __name__ == "__main__":
    # Quick demo
    shape = Shapes.TRIANGLE
    print(f"Shape has {shape.num_facets} facets")
    for i, indices in enumerate(shape.facet_indices):
        print(f"Face {i}: {'->'.join(map(str, indices))}")

    shape = Shapes.QUADRILATERAL
    print(f"Shape has {shape.num_facets} facets")
    for i, indices in enumerate(shape.facet_indices):
        print(f"Face {i}: {'->'.join(map(str, indices))}")
