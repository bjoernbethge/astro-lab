# Loader Module

Auto-generated documentation for `astro_lab.simulation.loader`

## Functions

### get_particle_type_num(part_type: str, sim_type: str = 'illustris') -> int

Convert particle type name to number.

Args:
    part_type: Particle type name
    sim_type: Simulation type

Returns:
    Particle type number

### get_particle_type_num_local(part_type: str, sim_type: str = 'illustris') -> int

Convert particle type name to number.

Args:
    part_type: Particle type name
    sim_type: Simulation type

Returns:
    Particle type number

### load_eagle(base_path: Union[str, pathlib.Path], snap_num: int, part_types: Union[str, List[str]] = ['gas', 'stars'], **kwargs) -> Dict[str, Any]

Load EAGLE snapshot.

Args:
    base_path: Base simulation directory
    snap_num: Snapshot number
    part_types: Particle types to load
    **kwargs: Additional arguments

Returns:
    Dictionary with simulation data

### load_illustris(base_path: Union[str, pathlib.Path], snap_num: int, part_types: Union[str, List[str]] = ['gas', 'stars'], depth: int = 8, index_path: Optional[pathlib.Path] = None, index_method: str = 'mesh', adaptive: bool = False, gpu: bool = False, cosmology: Optional[Dict[str, float]] = None) -> Dict[str, Any]

Load IllustrisTNG snapshot with enhancements.

Args:
    base_path: Base path (usually ending in 'output')
    snap_num: Snapshot number
    part_types: Particle types to load
    depth: Mesh depth for indexing
    index_path: Path for index cache
    index_method: Indexing method
    adaptive: Use adaptive mesh
    gpu: Use GPU acceleration
    cosmology: Cosmological parameters

Returns:
    Dictionary with simulation data

### load_simba(base_path: Union[str, pathlib.Path], snap_num: int, part_types: Union[str, List[str]] = ['gas', 'stars'], **kwargs) -> Dict[str, Any]

Load SIMBA snapshot.

Args:
    base_path: Base simulation directory
    snap_num: Snapshot number
    part_types: Particle types to load
    **kwargs: Additional arguments

Returns:
    Dictionary with simulation data

### load_simulation(base_path: Union[str, pathlib.Path], snap_num: int, part_types: Union[str, List[str]] = ['gas', 'stars'], sim_type: str = 'auto', **kwargs) -> Dict[str, Any]

Universal loader that auto-detects simulation type.

Args:
    base_path: Base simulation directory
    snap_num: Snapshot number
    part_types: Particle types to load
    sim_type: Simulation type or 'auto'
    **kwargs: Additional arguments

Returns:
    Dictionary with simulation data

## Classes

### SimulationLoader

Unified loader for different simulation formats.

### TNG50Loader

Specialized loader for TNG50/IllustrisTNG data.

Tested with TNG50-4 snapshots.

#### Methods

**`load_snapshot(snap_file: Union[str, pathlib.Path], particle_types: List[str] = ['PartType5'], max_particles: Optional[int] = None) -> Dict[str, Any]`**

Load a single TNG50 snapshot file.

Args:
snap_file: Path to HDF5 snapshot file
particle_types: Particle types to load
max_particles: Max particles per type

Returns:
Dictionary with loaded data
