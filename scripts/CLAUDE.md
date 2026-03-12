# gmsh-tests

Scripts for generating unstructured meshes with Gmsh and visualising them with meshio + PyVista.

## Key files

- `main.py` — generates a cylinder mesh using the Gmsh OpenCASCADE kernel, writes to `.msh`
- `visualise.py` — visualises a mesh file with PyVista; supports clipped and whole-cell half-space views

## Environment

- Package manager: **uv** — use `uv run <script>` to execute scripts
- Python: >=3.14
- Linting/formatting: **ruff**
- Type checking: **pyrefly**
- Formatting: always run the **ruff** formatter after writing code.
- Formatting: always use **ruff** to organise imports after new imports are added
- Type hints: always add type hints to parameters and return values.

## Conventions

- Always explain the aim of a proposed change before making it.
- Gmsh models must have at least one physical group defined, or meshio will warn about untagged cells.
- Cylinder axis is along the x-axis, centred at the origin.
