import math
from collections import deque, namedtuple
from dataclasses import dataclass
from itertools import islice, cycle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from gdtk import lmr

from cellshapes import Shapes

if TYPE_CHECKING:
    from typing import Self

    from gdtk.geom.sgrid import StructuredGrid

REPO_ROOT = Path("/home/alex/GDTk/gdtk.pythonic-dataclasses/")
MAX_VERTICES = 4
SENTINAL = -1


class IndexDict(dict):
    _count: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set the count correct from init
        self._count = len(self)

    def __getitem__(self, key):
        # Use internal counter to avoid problems with del or pop
        if key in self:
            return super().__getitem__(key)

        new_index = self._count
        super().__setitem__(key, new_index)
        self._count += 1
        return new_index

    def tolist(self):
        pairs = sorted(self.items(), key=lambda pair: pair[1])
        return list(map(lambda pair: pair[0], pairs))

    def __setitem__(self, key, value):
        raise NotImplementedError("IndexDict entries are immutable")


def sliding_window(iterable, n):
    "Collect data into overlapping fixed-length chunks or blocks."
    # sliding_window('ABCDEFG', 4) → ABCD BCDE CDEF DEFG
    iterator = iter(iterable)
    window = deque(islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)


def construct_halfspace(
    spanning_vertices: np.ndarray, positive_dir: np.ndarray | None = None
) -> (np.ndarray, float):
    # Assumes the vertices minimally span the dividing (hyper-)plane
    # Here we assume that each *row* is a vertex / point
    # We care about the *orientation*, so let's make everything
    # relative to v0.

    vertices = spanning_vertices.copy()
    # Because of the cholesky decomposition, the last vector will
    # be changed the most, while the first is preserved.
    # So, we remove the *last* vertex, and replace it with ones.
    anchor = vertices[-1, :].copy()
    vertices -= anchor

    if positive_dir is None:
        positive_dir = np.ones_like(anchor)

    vertices[-1, :] = positive_dir

    decomp = np.linalg.cholesky(vertices @ vertices.T)
    basis = np.linalg.solve(decomp, vertices)
    # Trim out numbers that are too small and might cause errors
    basis[np.abs(basis) < 1e3 * np.finfo(basis.dtype).eps] = 0.0

    normal = basis[-1, :]
    offset = np.vecdot(normal, anchor)

    return normal, offset


@dataclass
class Grid:
    geometry: GridGeometry
    topology: GridTopology

    @classmethod
    def from_structured_grid(cls, sgrid: StructuredGrid) -> Self:
        # We assume the structured grid will always be HEX cells

        Shape = namedtuple("Shape", "i, j, k")
        vert_grid_shape = Shape(i=sgrid.niv, j=sgrid.njv, k=sgrid.nkv)
        cell_grid_shape = Shape(i=sgrid.niv - 1, j=sgrid.njv - 1, k=max(sgrid.nkv - 1, 1))

        num_verts = math.prod(vert_grid_shape)
        num_cells = math.prod(cell_grid_shape)

        match sgrid.dimensions:
            case 2:
                cell = Shapes.QUADRILATERAL

                num_dims = 2
                vertices_per_facet = 2
                num_facets = 2 * num_cells + cell_grid_shape.i + cell_grid_shape.j

                vertex_coordinates = np.stack(
                    (sgrid.vertices.x, sgrid.vertices.y), axis=-1
                ).reshape((num_verts, cell.dimension))

            case 3:
                cell = Shapes.HEXAHEDRON

                num_dims = 3
                vertices_per_facet = 4
                num_facets = (
                    3 * num_cells
                    + cell_grid_shape.i * cell_grid_shape.j
                    + cell_grid_shape.j * cell_grid_shape.k
                    + cell_grid_shape.k * cell_grid_shape.i
                )

                vertex_coordinates = np.stack(
                    (sgrid.vertices.x, sgrid.vertices.y, sgrid.vertices.z), axis=-1
                ).reshape((num_verts, cell.dimension))
            case _:
                raise ValueError("Grid must have 2 or 3 dimensions")

        # Anti-clockwise from BOTTOM-LEFT
        vertex_offsets = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        # SOUTH, [FRONT,] EAST, NORTH, [BACK,], WEST
        # Same ordering as faces
        # quadrilateral version
        # (i, j, k) -> (x, y, z)
        neighbour_offsets = np.array(
            [
                [0, -1, 0],
                [+1, 0, 0],
                [0, +1, 0],
                [-1, 0, 0],
            ]
        )

        # [TODO]: For unstructured grids, replace with a `ragged` array in axis=1
        cell_vertices = np.full((num_cells, cell.num_vertices), fill_value=SENTINAL)
        cell_adjacency = np.full((num_cells, cell.num_facets), fill_value=SENTINAL)

        # Construct `cell_vertices` and `cell adjacency`
        for cell_idx, (i, j, k) in enumerate(np.ndindex(cell_grid_shape)):
            vertices = vertex_offsets + (i, j, k)
            vertex_idxs = np.ravel_multi_index(vertices.T, vert_grid_shape)
            cell_vertices[cell_idx, :] = vertex_idxs

            neighbours = neighbour_offsets + (i, j, k)
            neighbour_idxs = np.ravel_multi_index(neighbours.T, cell_grid_shape, mode="wrap")
            # We can remove this if we want to have "wrapped" domain
            # Neighbours that are outside [0, length) are invalid
            valid = np.all(
                np.logical_and(neighbours >= 0, neighbours < cell_grid_shape),
                axis=1,
            )
            neighbour_idxs[~valid] = SENTINAL
            cell_adjacency[cell_idx, :] = neighbour_idxs

        # [TODO]: For unstructured grids, replace with a `ragged` array in axis=1
        facet_vertices = np.full((num_facets, vertices_per_facet), fill_value=SENTINAL)
        cell_facets = np.full((num_cells, cell.num_facets), fill_value=SENTINAL)

        # Construct `facet_vertices`
        facet_id_lookup = IndexDict()
        for cell_idx in range(num_cells):
            # Store vertices in ascending order
            cell_facet_vertices = np.sort(cell_vertices[cell_idx, cell.facet_indices], axis=-1)
            cell_facet_ids = [facet_id_lookup[tuple(facet)] for facet in cell_facet_vertices]

            cell_facets[cell_idx, :] = cell_facet_ids
            facet_vertices[cell_facet_ids, :] = cell_facet_vertices

        assert len(facet_id_lookup) == num_facets, "Predicted number of facets is incorrect."

        # Determine which two (2) cells each facet is connecting?
        for cell_idx in range(num_cells):
            # Assume that the order of cell_facets is the same as cell_adjacency
            # i.e. the face description is the same
            neighbour_idxs = cell_adjacency[cell_idx, :]
            is_facet_outwards = neighbour_idxs < cell_idx  # Higher -> lower
            facet_sign = np.where(is_facet_outwards, +1, -1)
            cell_facets[cell_idx, :] *= facet_sign
            print(cell_facets[cell_idx, :])

        for facet_id in range(num_facets):
            vertex_coordinates[facet_vertices[facet_id, :], :]

        # geometry = GridGeometry(vertex_coordinates=vertex_coordinates, cell_half_spaces=)
        # topology = GridTopology(cell_vertices=cell_vertices, cell_adjacency=cell_adjacency)

        # return Grid(geometry=geometry, topology=topology)


@dataclass
class GridGeometry:
    """Stores the geometric / positional data as cartesian coordinates."""

    # For v vertices, e edges, f faces, and c cells
    # For ~n facets per face (d+1 if using simplices)

    # Must be known / constructed from StructuredGrid
    vertices: np.ndarray  # Coordinates (v,d)

    # Constructed from geom.vertices & topo.cell_vertices
    # Is it worth storing these? Yes for now, let's see how much we used them.
    cell_centers: np.ndarray  # Coordinates (c,d)

    # Constructed from geom.vertices & topo.facet_vertices & topo.cell_vertices & topo.cell_adjacency?
    facet_hyperplanes: np.ndarray  # Normal vector & offset -> (f,d+1)


@dataclass
class GridTopology:
    """Stores the topological / relational data as indices to relevant geometry data."""

    # [NOTE]: Topology is very *ragged* currently

    # For v vertices, e edges, and f faces
    # For ~n vertices per face (d+1 if using simplices)

    # Must be known / constructed from StructuredGrid
    cell_vertices: np.ndarray  # Vertex-ids (c,~n?)

    # Should be known / constructed from StructuredGrid?
    # Otherwise can be computed from cell_vertices (expensive)
    cell_facets: np.ndarray  # Cell-to-face connections (c,?)

    # Should be known / constructed from StructuredGrid?
    # Otherwise can be computed from cell_vertices (expensive)
    cell_adjacency: np.ndarray  # Face-to-face connections (f,~?)

    # [NOTE]: We're getting some redundancy here. The triplet of
    # (this, cell_adjacency, cell_facets) is redundant, one can
    # always be determined from the other two.
    # Hmm, probably not great to use this, since currently we don't have
    # separate indices for each boundary. That means cell-id 0 (bottom-left
    # corner) will match to -1 for both WEST & SOUTH.
    cell_connections: np.ndarray  # Cell-to-cell connections (c*f/2,2)

    # Should be known / constructed from StructuredGrid?
    # Otherwise can be computed from cell_vertices (expensive)
    # [NOTE]: Is the dual to cell_adjacency, can be done at the same time
    # Potentially we can only store d vertices per facet.
    facet_verties: np.ndarray  # Vertex-ids (f,~n?)


def main():
    lmr_cfg = lmr.LmrConfig(REPO_ROOT / "src/lmr/lmr.cfg")
    sim_data = lmr.SimInfo(lmr_cfg)
    snap = sim_data.read_snapshot("0001")
    grid = snap.grids[0]

    Grid.from_structured_grid(grid)


if __name__ == "__main__":
    main()
