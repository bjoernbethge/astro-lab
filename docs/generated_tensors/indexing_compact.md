# Indexing Module

Auto-generated documentation for `astro_lab.simulation.indexing`

## Classes

### BaseIndex

Abstract base class for spatial indices.

#### Methods

**`build(self) -> None`**

Build the index structure.

**`query_box(self, bounds: numpy.ndarray, return_distances: bool = False) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]`**

Query particles in box region.

**`query_sphere(self, center: numpy.ndarray, radius: float, return_distances: bool = False) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]`**

Query particles in spherical region.

**`save(self, path: pathlib.Path) -> None`**

Save index to disk.

**`load(self, path: pathlib.Path) -> None`**

Load index from disk.

### HierarchicalIndex

Hierarchical spatial index using adaptive octree structure.

Automatically refines in high-density regions for optimal
performance with clustered data.

#### Methods

**`build(self) -> None`**

Build hierarchical index.

**`query_box(self, bounds: numpy.ndarray, return_distances: bool = False) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]`**

Query particles in box region.

**`query_sphere(self, center: numpy.ndarray, radius: float, return_distances: bool = False) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]`**

Query particles in spherical region.

**`save(self, path: pathlib.Path) -> None`**

Save index to HDF5 file.

**`load(self, path: pathlib.Path) -> None`**

Load index from HDF5 file.

### IndexFactory

Factory for creating appropriate index types.

#### Methods

**`create_index(positions: numpy.ndarray, box_size: float, index_type: str = 'auto', **kwargs) -> astro_lab.simulation.indexing.BaseIndex`**

Create spatial index of specified type.

Args:
positions: Particle positions
box_size: Box size
index_type: Type of index ('hierarchical', 'hash', 'octree', 'auto')
**kwargs: Additional arguments for index constructor

Returns:
Spatial index instance

### IndexNode

Node in hierarchical index structure.

### OctreeIndex

Classic octree implementation with fixed splitting.

More memory efficient than adaptive hierarchical index
but less flexible.

#### Methods

**`get_depth_statistics(self) -> Dict[int, int]`**

Get number of nodes at each depth level.

### SpatialHashIndex

Spatial hash index for uniform particle distributions.

Very fast for uniformly distributed data but less efficient
for highly clustered distributions.

#### Methods

**`build(self) -> None`**

Build spatial hash table.

**`query_box(self, bounds: numpy.ndarray, return_distances: bool = False) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]`**

Query particles in box region.

**`query_sphere(self, center: numpy.ndarray, radius: float, return_distances: bool = False) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]`**

Query particles in spherical region.

**`save(self, path: pathlib.Path) -> None`**

Save index to file.

**`load(self, path: pathlib.Path) -> None`**

Load index from file.
