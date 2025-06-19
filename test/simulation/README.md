# Simulation Tests

This directory contains tests for the `astro_lab.simulation` module, split into focused test files for better organization and maintainability.

## Test Files

- `test_simulation_tensor.py` - Tests for `SimulationTensor` tensor-based simulation architecture
- `test_cosmology_calculator.py` - Tests for `CosmologyCalculator` cosmological calculations
- `test_visualization_integration.py` - Tests for visualization integration (PyVista/Blender)
- `test_integration.py` - Integration tests for the complete tensor-based system

## Key Features Tested

### SimulationTensor
- Basic initialization with positions, features, and edges
- Periodic boundary conditions and distance calculations
- Particle subset operations and center of mass calculations
- PyTorch Geometric integration
- Redshift updates and cosmological metadata

### CosmologyCalculator
- Hubble parameter calculations with tensor support
- Comoving, angular diameter, and luminosity distances
- Age of universe calculations
- Cosmological consistency checks

### Visualization Integration
- PyVista mesh conversion for 3D visualization
- Blender integration for advanced rendering
- Memory information and optimization

### Integration Tests
- Complete TNG50 workflow simulation
- Cross-component compatibility testing

## Running Tests

Run all simulation tests:
```bash
pytest test/simulation/
```

Run specific test file:
```bash
pytest test/simulation/test_simulation_tensor.py
```

Run with specific markers:
```bash
pytest test/simulation/ -m "not slow"  # Skip slow tests
pytest test/simulation/ -k "cosmology"  # Only cosmology tests
```

## Notes

- Tests handle different return types (torch.Tensor, numpy.ndarray, float) for robust compatibility
- PyVista and Blender tests are skipped if dependencies are not available
- Integration tests use realistic TNG50-like parameters for comprehensive testing 