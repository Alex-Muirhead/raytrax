= RaytraX

This codebase attempts to implement *volumetric raytracing for a participating and emitting fluid medium*, using a monte-carlo integration of the radiation transport equation, calculated through ray-tracing.

== Domain Assumptions

The domain is assumed to be presented in the form of an unstructured grid from a `.vtk` file.
Properties of the fluid are all cell-centered. The relevant properties for radiation transport are the chemical composition and temperatures (i.e. electronic temperature, and temperature for each species) at each cell.
When the result is computed, the radiative heating (divergence of radiative heat flux) should be written to a duplicated `.vtk` file.

The domain may either be 2D or 3D when parsed in the file.
In the case of a 3D grid, the operation is umambigous.
In the case of a 2D grid, the operation may either by "planar", or "axissymetric". 
  / planar: The domain is assumed to extend infinitely in the z direction (assuming the domain is xy).
    Each ray direction is still chosen from S2 (in 3D), however the direction is projected to the 2D domain, and a cosine direction factor is used to correct the distance traversed in 2D to the actual 3D distance.
  / axissymetric: The domain is assumed to be rotated around the x-axis (assuming the domain is xy).
    Each ray direction is still chosen from S2 (in 3D), however the ray is now projected to 2D, and turns from a _linear_ path to a _hyperbolic_ path. The hyperbolic path can be represented by the two linear paths of the asymptotes, and the correct intersections can be found from intersections with the asymptotes.

For the purposes of the radiation transport, values are assumed to be constant across each cell, and no attempt at reconstruction of in-cell variation of properties is to be made.

Each cell is assumed to be one of the 1st order VTK cells
  / 2D: Either `VTK_TRIANGLE` or `VTK_QUAD`
  / 3D: Any of `VTK_TETRA`, `VTK_HEXAHEDRON` (same as `VTK_VOXEL`), `VTK_WEDGE`, or `VTK_PYRAMID`
The mesh must be exclusively 2D or 3D, but can contain a mixture of elements from within that category.
From this restriction, every cell within the mesh is convex, and can be represented in the half-space form.

The entire mesh will be simply connected.

When applying boundary conditions, these will be applied to face groups by their tag.
Face groups will not need to be decomposed in order to apply boundary conditions.

== Method

=== Geometry

Rays are sampled from within the spatial domain of the mesh randomly, where the probability function of selection is proportional to the radiative energy emitted at that point.
Consequently, the origin of the ray is first chosen by cell (as each cell has assumed constant properties) according to the radiative energy emitted by the cell, and then randomly within the cell with uniform distribution.
The emission of radiation is assumed to be isotropic in direction, and as such the direction of the ray is chosen from S2 with uniform distribution.

Rays will be marched across the domain geometrically. When created, a cell will know
+ The current spatial point (the origin)
+ The direction
+ The containing cell (from generation)
As each cell is convex, crossing the cell becomes a contained convex problem.
With each cell represented as half-spaces, this becomes a simple "max-distance" within a convex domain, with the constraint that the distance travelled must be positive (explicitly >0, or > eps), so that the current location isn't used as a solution.
The face which the cell crosses is used in conjunction with the cell connectivity to determine the next contained cell.

After each cell, the cell will know
+ The current spatial point (the previous spatial point + distance travelled $times$ distance)
+ The direction (unchanged)
+ The containing cell (from the connectivity and exit face)

By inductive reasoning, the cell will always know where it is.

The cell connectivity should be constructed such that the boundary faces of the domain connect to the boundary cell and the "exterior cell" (represented by a sentinal value). The exterior cell/s should not be explicitly stored anywhere in the geometry of the mesh. When a ray considers it's current cell an exterior cell, it has exited the domain, and will be terminated.
The exact sentinal value may be used to decide what happens upon termination.

=== Radiation Transport

The transport of radiative energy can happen in two operational modes:

/ Forwards: Each ray carries energy from the origin cell, and deposits the energy along the path.
  This can either happen
    / discretely: where the entire energy is deposited into a single cell.
      This cell is selected along the length of the ray, by choosing an "optical distance" from an exponential distribution.
      The optical distance is given by the integral of the absorptivity $kappa$ over the distance travelled.
      Once the ray travels this selected optical distance, all the energy of the ray is deposited into the current cell.
    / continuously: where the energy is deposited among all cells along the ray.
      The energy is deposited according to the exponential of the optical distance travelled in each cell.
      If the distance travelled in a cell is $s$, then the energy deposited is $q = q_0 (1 - exp(-kappa s))$.
      The energy remaining in the ray after this cell is $q_0 exp(-kappa s)$.

/ Backwards: Each ray accumulates energy from cells it crosses (according to the emitted power of said cell), along with the optical distance the ray has travelled. The accumulated energy from each cell is multiplied by the decay factor from the respective cumulative optical distance. Once the ray has terminated, the accumulated (weighted) energy is deposited into the origin cell.

== Stages of implementation

+ A simple unstructured mesh (cylindrical geometry), with constant temperature. The gas is modelled as a "black-body radiator", such that the emissivity ($kappa$)---and consequently absorptivity---is constant. The heat-flux (radiative divergence) from a volume of constant temperature gas is given by the Planck mean:
  $ nabla dot arrow(q)_"rad" = 4 kappa sigma T^4 $
  where $sigma$ is the Boltzmann constant.

  No frequency dependence is necessary for any calculation. The value for $kappa$ is chosen arbitrarily.
  Continous forward operation mode is used.

+ The cylinder temperature is now changed. Split by the z-value, half the cylinder is made hot, and half cold. This should demonstrate the drop-off of radiative heating in the cold region, in the direction away from the splitting z-value. When considering the heating in the cold region along, along the central axis it should have a heating profile similar to that of tangent slab.

+ The "hot-half" setup should be replaced by a boundary condition on one face of the cylinder. Faces on this boundary will have the same "hot" temperature value as in the prior stage. When selecting the origin of rays, they should have a chance of starting on the boundary faces, with the probabilities equal to the emitted energy of the face (analogous to the emitted energy of each cell). The units of these energies (between cells and faces) must be equal to allow for comparison.
