import gmsh
from tap import Tap


def create_plane_mesh(
    width: float = 1.0,
    height: float = 1.0,
    output: str = "plane.msh",
):
    """Generate an unstructured mesh of a plane.

    :param width: Plane radius
    :param height: Plane length along x-axis
    """
    gmsh.initialize()
    gmsh.model.add("plane")

    # Cylinder centered at origin, axis along x-axis
    # gmsh.model.occ.addCylinder(x, y, z, dx, dy)
    # Start point at (-length/2, 0, 0), direction along x
    face_tag = gmsh.model.occ.addRectangle(-width / 2, -height / 2, 0, width, height, 0)

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(dim=2, tags=[face_tag], name="plane")

    # gmsh.option.set_number("Mesh.Algorithm", 11)
    # gmsh.option.set_number("Mesh.MeshSizeMin", width / 2)

    gmsh.model.mesh.generate(dim=2)

    gmsh.write(output)
    gmsh.finalize()
    print(f"Mesh written to {output}")


class Args(Tap):
    """Generate an unstructured mesh of a plane."""

    width: float = 1.0  # Cylinder radius
    height: float = 1.0  # Cylinder length along x-axis
    output: str = "plane.msh"  # Output file (format inferred from extension)


def main():
    args = Args(underscores_to_dashes=True).parse_args()

    create_plane_mesh(
        width=args.width,
        height=args.height,
        output=args.output,
    )


if __name__ == "__main__":
    main()
