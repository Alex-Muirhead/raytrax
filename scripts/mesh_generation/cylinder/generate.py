import gmsh
from tap import Tap


def create_cylinder_mesh(
    radius: float = 1.0,
    length: float = 1.0,
    mesh_size: float = 0.1,
    output: str = "cylinder.msh",
):
    """Generate an unstructured mesh of a cylinder.

    :param radius: Cylinder radius
    :param length: Cylinder length along x-axis
    :param mesh_size: Mesh size factor
    """
    gmsh.initialize()
    gmsh.model.add("cylinder")

    # Cylinder centered at origin, axis along x-axis
    # gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, r)
    # Start point at (-length/2, 0, 0), direction along x
    vol_tag = gmsh.model.occ.addCylinder(-length / 2, 0, 0, length, 0, 0, radius)

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(dim=3, tags=[vol_tag], name="cylinder")

    gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size)

    gmsh.model.mesh.generate(dim=3)

    gmsh.write(output)
    gmsh.finalize()
    print(f"Mesh written to {output}")


class Args(Tap):
    """Generate an unstructured mesh of a cylinder."""

    radius: float = 1.0  # Cylinder radius
    length: float = 1.0  # Cylinder length along x-axis
    mesh_size: float = 0.1  # Mesh size factor
    output: str = "cylinder.msh"  # Output file (format inferred from extension)


def main():
    args = Args(underscores_to_dashes=True).parse_args()

    create_cylinder_mesh(
        radius=args.radius,
        length=args.length,
        mesh_size=args.mesh_size,
        output=args.output,
    )


if __name__ == "__main__":
    main()
