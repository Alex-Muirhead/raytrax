import pyvista as pv
from tap import Tap


def visualise_cross_section(mesh_file: str, whole_cells: bool = False) -> None:
    mesh = pv.read(mesh_file)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color="lightblue", edge_color="gray")
    plotter.add_axes()
    plotter.show()


class Args(Tap):
    """Visualise a y=0 cross-section of a mesh using PyVista."""

    mesh_file: str  # Path to the mesh file (e.g. cylinder.msh, cylinder.vtk)
    whole_cells: bool = False  # Show all cells with at least one vertex at y<=0, rather than clipping cells at y=0.

    def configure(self) -> None:
        self.add_argument("mesh_file")  # Makes it positional


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    visualise_cross_section(args.mesh_file, whole_cells=args.whole_cells)


if __name__ == "__main__":
    main()
