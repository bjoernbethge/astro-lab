# Mesh Module

Auto-generated documentation for `astro_lab.simulation.mesh`

## Functions

### create_mesh(positions: numpy.ndarray, box_size: float, depth: int = 8, adaptive: bool = False, gpu: bool = False, **kwargs) -> Union[astro_lab.simulation.mesh.AstroMesh, astro_lab.simulation.mesh.AdaptiveMesh]

Factory function to create appropriate mesh.

Args:
    positions: Particle positions
    box_size: Box size
    depth: Mesh depth
    adaptive: Use adaptive refinement
    gpu: Use GPU acceleration
    **kwargs: Additional arguments

Returns:
    Mesh instance

## Classes

### AdaptiveMesh

Adaptive mesh refinement for non-uniform particle distributions.

Automatically refines mesh in high-density regions for better
performance with clustered astronomical data.

#### Methods

**`build(self) -> Tuple[numpy.ndarray, numpy.ndarray]`**

Build adaptive spatial index.

Returns:
Tuple of (rank, mark) arrays

**`query_adaptive(self, boundary: numpy.ndarray, max_level: Optional[int] = None) -> numpy.ndarray`**

Query with adaptive refinement awareness.

Args:
boundary: Query box
max_level: Maximum refinement level to consider

Returns:
Particle indices

### AstroMesh

Enhanced mesh for astronomical spatial indexing.

Improvements over original:
- GPU acceleration option
- Better memory efficiency
- Support for different coordinate systems
- Periodic boundary handling

#### Methods

**`build(self) -> Tuple[numpy.ndarray, numpy.ndarray]`**

Build spatial index.

Returns:
Tuple of (rank, mark) arrays

**`query_box(self, boundary: numpy.ndarray, method: str = 'outer') -> numpy.ndarray`**

Query particles in box region.

Args:
boundary: Box boundaries [[xmin,ymin,zmin], [xmax,ymax,zmax]]
method: Query method ('outer', 'inner', 'exact')

Returns:
Particle indices in box
