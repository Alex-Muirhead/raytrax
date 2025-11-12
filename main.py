import math
from collections import deque, namedtuple
from dataclasses import dataclass
from itertools import islice, cycle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from gdtk import lmr

if TYPE_CHECKING:
    from typing import Self

    from gdtk.geom.sgrid import StructuredGrid

REPO_ROOT = Path("/home/alex/GDTk/gdtk.robust-python-modules")
MAX_VERTICES = 4
SENTINAL = -1


def sliding_window(iterable, n):
    "Collect data into overlapping fixed-length chunks or blocks."
    # sliding_window('ABCDEFG', 4) → ABCD BCDE CDEF DEFG
    iterator = iter(iterable)
    window = deque(islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)


def construct_halfspace(spanning_vertices: np.ndarray) -> (np.ndarray, float):
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
    vertices[-1, :] = np.ones_like(anchor)

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
                num_dims = 2
                facets_per_cell = 4
                vertices_per_cell = 4

                vertex_coordinates = np.stack(
                    (sgrid.vertices.x, sgrid.vertices.y), axis=-1
                ).reshape((num_verts, num_dims))

            case 3:
                num_dims = 3
                facets_per_cell = 6
                vertices_per_cell = 8

                vertex_coordinates = np.stack(
                    (sgrid.vertices.x, sgrid.vertices.y, sgrid.vertices.z), axis=-1
                ).reshape((num_verts, num_dims))
            case _:
                raise ValueError("Grid must have 2 or 3 dimensions")

        # SOUTH, [FRONT,] EAST, NORTH, [BACK,], WEST
        neighbour_offsets = np.array([[-1, 0, 0], [0, +1, 0], [+1, 0, 0], [0, -1, 0]])
        # Anti-clockwise from BOTTOM-LEFT
        vertex_offsets = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

        cell_vertices = np.full((num_cells, vertices_per_cell), fill_value=SENTINAL)
        cell_adjacency = np.full((num_cells, facets_per_cell), fill_value=SENTINAL)

        # We include the dual-edges to the "exterior cell"
        num_dual_edges = 2 * num_cells + cell_grid_shape.i + cell_grid_shape.j
        dual_edges = np.full((num_dual_edges, 2), fill_value=SENTINAL)
        dual_edge_lookup = dict()

        cell_connection_list = np.full((num_cells, facets_per_cell), fill_value=SENTINAL)

        edge_counter = 0
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

            local_connections = []
            local_vertex_path = sliding_window(cycle([0, 1, 2, 3]), 2)  # Loop first element/s

            for neighbour, facet_span in zip(neighbour_idxs, local_vertex_path):
                dual_edge = (cell_idx, int(neighbour))

                # Check if the (flipped) dual-edge already exists first
                # Yes, I could calculate this. It's a pain, this is easier.
                edge_idx = dual_edge_lookup.get(dual_edge[::-1], edge_counter)
                local_connections.append(edge_idx)

                if edge_idx != edge_counter:
                    # The edge did already exist
                    # No need to repeat the definitions
                    continue

                facet_normal, facet_offset = construct_halfspace(
                    vertex_coordinates[vertex_idxs[list(facet_span)], :]
                )

                print(facet_normal, facet_offset)

                # Otherwise, define our edge
                dual_edges[edge_idx] = dual_edge
                dual_edge_lookup[dual_edge] = edge_idx
                edge_counter += 1

            cell_connection_list[cell_idx, :] = local_connections

        print("Vertex shape: ", vert_grid_shape)
        # geometry = GridGeometry(vertex_coordinates=vertex_coordinates, cell_half_spaces=)
        # topology = GridTopology(cell_vertices=cell_vertices, cell_adjacency=cell_adjacency)

        # return Grid(geometry=geometry, topology=topology)


@dataclass
class GridGeometry:
    # For v vertices, e edges, f faces, and c cells
    # For ~n facets per face (d+1 if using simplices)
    vertex_coordinates: np.ndarray  # Coordinates (v,d)
    facet_hyperplanes: np.ndarray  # Normal vector & offset -> (f,d+1)


@dataclass
class GridTopology:
    # For v vertices, e edges, and f faces
    # For ~n vertices per face (d+1 if using simplices)
    cell_vertices: np.ndarray  # Vertex-ids
    cell_adjacency: np.ndarray  # Face-to-face connections (f,~n)


def main():
    lmr_cfg = lmr.LmrConfig(REPO_ROOT / "src/lmr/lmr.cfg")
    sim_data = lmr.SimInfo(lmr_cfg)
    snap = sim_data.read_snapshot("0001")
    grid = snap.grids[0]

    Grid.from_structured_grid(grid)


if __name__ == "__main__":
    main()
