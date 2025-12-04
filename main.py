from jax._src.interpreters.partial_eval import convert_constvars_jaxpr
from typing import overload
import math
from collections import deque, namedtuple
from dataclasses import dataclass
from itertools import islice
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


class IndexDict(dict[tuple[int, int], int]):
    _count: int

    def __init__(self, start: int = 0):
        super().__init__()
        # Set the count to start value
        self._count = start

    def __getitem__(self, key: tuple[int, int]) -> int:
        # Use internal counter to avoid problems with del or pop
        if key in self:
            return super().__getitem__(key)

        new_index = self._count
        super().__setitem__(key, new_index)
        self._count += 1
        return new_index

    def tolist(self) -> list[tuple[int, int]]:
        pairs = sorted(self.items(), key=lambda pair: pair[1])
        return list(map(lambda pair: pair[0], pairs))

    def __setitem__(self, key: tuple[int, int], value: int) -> int:
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

        # We need a 0th indexed "dummy" facet, since all our indices will be >=1, to
        # allow for the sign to be used as an indicator of direction
        num_facets += 1

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
        cell_center = np.empty((num_cells, num_dims))

        # Construct `cell_vertices`, `cell_center`, and `cell adjacency`
        for cell_idx, (i, j, k) in enumerate(np.ndindex(cell_grid_shape)):
            vertices = vertex_offsets + (i, j, k)
            vertex_idxs = np.ravel_multi_index(vertices.T, vert_grid_shape)
            cell_vertices[cell_idx, :] = vertex_idxs
            cell_center[cell_idx, :] = np.mean(vertex_coordinates[vertex_idxs, :], axis=0)

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
        # Start lookup at 1, so that we can use the sign
        facet_id_lookup = IndexDict(start=1)
        for cell_idx in range(num_cells):
            # Store vertices in ascending order
            cell_facet_vertices = np.sort(cell_vertices[cell_idx, cell.facet_indices], axis=-1)
            cell_facet_ids = [facet_id_lookup[tuple(facet)] for facet in cell_facet_vertices]

            cell_facets[cell_idx, :] = cell_facet_ids
            # A bit messy, but we need to account for the 1-offset
            facet_vertices[cell_facet_ids, :] = cell_facet_vertices

        # Ensure we count the "dummy" facet at index 0
        assert len(facet_id_lookup) + 1 == num_facets, "Predicted number of facets is incorrect."

        facet_cells = np.full((num_facets, 2), fill_value=SENTINAL)

        for cell_idx in range(num_cells):
            # Assume that the order of cell_facets is the same as cell_adjacency
            # i.e. the face description is the same
            neighbour_idxs = cell_adjacency[cell_idx, :]
            is_facet_outwards = neighbour_idxs < cell_idx  # Higher -> lower
            facet_sign = np.where(is_facet_outwards, +1, -1)
            cell_facets[cell_idx, :] *= facet_sign
            # A bit of redundancy here, but we just sort at the end
            facet_idxs = cell_facets[cell_idx, :]
            facet_cells[facet_idxs, 0] = cell_idx
            facet_cells[facet_idxs, 1] = neighbour_idxs

        # Sort now, to ensure cells go from Higher [0] -> Lower [1]
        facet_cells = np.fliplr(np.sort(facet_cells, axis=-1))

        facet_normals = np.empty((num_facets, num_dims))
        facet_offsets = np.empty((num_facets,))

        # Don't do anything for our "dummy" facet
        for facet_id in range(1, num_facets):
            facet_coordinates = vertex_coordinates[facet_vertices[facet_id, :], :]
            origin_cell = facet_cells[facet_id, 0]
            outwards_direction = np.mean(facet_coordinates, axis=0) - cell_center[origin_cell, :]
            normal, offset = construct_halfspace(facet_coordinates, outwards_direction)

            facet_normals[facet_id, :] = normal
            facet_offsets[facet_id] = offset

        geometry = GridGeometry(
            vertex_coordinates=vertex_coordinates,
            cell_centers=cell_center,
            facet_normals=facet_normals,
            facet_offsets=facet_offsets,
        )
        topology = GridTopology(
            cell_vertices=cell_vertices,
            cell_adjacency=cell_adjacency,
            cell_facets=cell_facets,
            facet_cells=facet_cells,
            facet_verties=facet_vertices,
        )

        return Grid(geometry=geometry, topology=topology)


@dataclass
class GridGeometry:
    """Stores the geometric / positional data as cartesian coordinates."""

    # For v vertices, e edges, f faces, and c cells
    # For ~n facets per face (d+1 if using simplices)

    # Must be known / constructed from StructuredGrid
    vertex_coordinates: np.ndarray  # Coordinates (v,d)

    # Storing them for now, maybe we can replace this info later
    cell_centers: np.ndarray

    # Constructed from geom.vertices & topo.facet_vertices & topo.cell_vertices & topo.cell_adjacency?
    # Maybe pack together info (f,d+1) for cache efficiency later?
    facet_normals: np.ndarray  # Normal vector -> (f,d)
    facet_offsets: np.ndarray  # Offset -> (f,)


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
    facet_cells: np.ndarray  # Face-to-face connections (f,~?)

    # [NOTE]: We're getting some redundancy here. The triplet of
    # (this, cell_adjacency, cell_facets) is redundant, one can
    # always be determined from the other two.
    # Hmm, probably not great to use this, since currently we don't have
    # separate indices for each boundary. That means cell-id 0 (bottom-left
    # corner) will match to -1 for both WEST & SOUTH.
    cell_adjacency: np.ndarray  # Cell-to-cell connections (c*f/2,2)

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

    mygrid = Grid.from_structured_grid(grid)

    # Starting vector at center of cell id=0
    cell_id = 0
    while cell_id != -1:
        p = mygrid.geometry.cell_centers[cell_id, :]
        q = np.abs(np.random.rand(2))  # Ensure we go north-east for now
        q /= np.linalg.norm(q)  # Normalise

        facet_ids = mygrid.topology.cell_facets[cell_id, :]
        facet_signs, facet_ids = np.sign(facet_ids), np.abs(facet_ids)
        # Add new axis to `facet_signs` to ensure broadcast is correct
        convex_normals = mygrid.geometry.facet_normals[facet_ids, :] * facet_signs[:, None]
        convex_offsets = mygrid.geometry.facet_offsets[facet_ids] * facet_signs

        # b - A@p
        t_all = (convex_offsets - convex_normals @ p) / (convex_normals @ q)
        t_all[t_all <= 0] = np.nan
        t = np.nanmin(t_all)
        idx = np.nanargmin(t_all)
        print(f"t = {t:.3f} at facet {idx}")
        print(f"p + tq = {p + t * q}")
        next_cell_id = mygrid.topology.cell_adjacency[cell_id, idx]
        print(f"The next cell is {next_cell_id}")
        cell_id = next_cell_id


if __name__ == "__main__":
    main()
