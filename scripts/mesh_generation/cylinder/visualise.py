import pyvista as pv
from pyvista import UnstructuredGrid
from tap import Tap


def visualise_cross_section(mesh_file: str, cut_cells: bool = False, scalars: str | None = None) -> None:
    mesh = pv.read(mesh_file)

    if cut_cells:
        # Clip at y=0, keeping the y<=0 half (cuts through cells at the boundary).
        half: UnstructuredGrid = mesh.clip(normal="y", origin=(0, 0, 0))
    else:
        # Select all cells that have at least one vertex with y <= 0,
        # so no cell is cut in half by the clipping plane.
        half: UnstructuredGrid = mesh.extract_points(mesh.points[:, 1] <= 0, adjacent_cells=True)

    if half.number_of_cells == 0:
        half = mesh

    if scalars is not None:
        half.set_active_scalars(scalars)

    plotter = pv.Plotter()
    plotter.add_mesh(half)
    plotter.add_axes()
    plotter.show()


class Args(Tap):
    """Visualise a y=0 cross-section of a mesh using PyVista."""

    mesh_file: str  # Path to the mesh file (e.g. cylinder.msh, cylinder.vtk)
    cut_cells: bool = False  # Show all cells with at least one vertex at y<=0, rather than clipping cells at y=0.
    scalars: str | None = None

    def configure(self) -> None:
        self.add_argument("mesh_file")  # Makes it positional


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    visualise_cross_section(args.mesh_file, cut_cells=args.cut_cells, scalars=args.scalars)


if __name__ == "__main__":
    main()
