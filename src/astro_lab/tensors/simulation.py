"""
Simulation Tensor for Cosmological Simulations
=============================================

Tensor for cosmological simulation data (TNG50, Illustris, etc.) with
integrated cosmology calculations and particle handling.

Combines spatial coordinates with simulation-specific metadata,
cosmological calculations, and particle type handling.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .base import AstroTensorBase

# Optional dependencies with fallbacks
try:
    import scipy.integrate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class SimulationTensor(AstroTensorBase):
    """
    Tensor for cosmological simulation data with integrated cosmology.
    
    Handles:
    - Particle positions, velocities, masses
    - Cosmological calculations (redshift, distances, age)
    - Simulation-specific metadata (box size, particle types)
    - Periodic boundary conditions
    - Graph structure for spatial relationships
    """

    _metadata_fields = [
        "simulation_name",      # "TNG50", "Illustris", etc.
        "particle_type",        # "gas", "dark_matter", "stars", "black_holes"
        "box_size",            # Simulation box size
        "redshift",            # Simulation redshift
        "scale_factor",        # Cosmological scale factor
        "cosmology",           # Cosmological parameters
        "snap_num",            # Snapshot number
        "hubble_param",        # Hubble parameter
        "time",                # Simulation time
        "mass_table",          # Mass table for particles
        "periodic_boundaries", # Whether boundaries are periodic
        "feature_names",       # Names of features in x tensor
    ]

    def __init__(
        self,
        positions: Union[torch.Tensor, np.ndarray],
        features: Optional[Union[torch.Tensor, np.ndarray]] = None,
        edge_index: Optional[torch.Tensor] = None,
        simulation_name: str = "TNG50",
        particle_type: str = "gas",
        box_size: float = 35000.0,  # TNG50 default in ckpc/h
        redshift: float = 0.0,
        cosmology: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Initialize simulation tensor.

        Args:
            positions: Particle positions [N, 3]
            features: Particle features [N, D] (mass, potential, etc.)
            edge_index: Graph edge indices [2, E] for spatial connections
            simulation_name: Name of simulation
            particle_type: Type of particles
            box_size: Simulation box size
            redshift: Redshift of snapshot
            cosmology: Cosmological parameters
            **kwargs: Additional metadata
        """
        # Convert to tensors
        pos_tensor = torch.as_tensor(positions, dtype=torch.float32)
        if pos_tensor.shape[-1] != 3:
            raise ValueError(f"Positions must have shape [..., 3], got {pos_tensor.shape}")

        # Store main data as positions with clean metadata
        super().__init__(
            pos_tensor,
            simulation_name=simulation_name,
            particle_type=particle_type,
            box_size=box_size,
            redshift=redshift,
            tensor_type="simulation",
            **kwargs,
        )

        # Store additional simulation data
        if features is not None:
            self._features = torch.as_tensor(features, dtype=torch.float32)
        else:
            self._features = None

        if edge_index is not None:
            self._edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        else:
            self._edge_index = None

        # Initialize cosmology
        if cosmology is None:
            cosmology = {"H0": 70.0, "Omega_m": 0.3, "Omega_Lambda": 0.7}
        self._cosmology = CosmologyCalculator(**cosmology)
        self.update_metadata(cosmology=cosmology)

        # Calculate derived quantities
        self._update_cosmological_metadata()

        logger.info(f"🌌 SimulationTensor created: {simulation_name} {particle_type}")
        logger.info(f"   {len(positions)} particles, z={redshift:.3f}, box={box_size:.1f}")

    def _update_cosmological_metadata(self) -> None:
        """Update cosmological metadata based on redshift."""
        z = self.get_metadata("redshift", 0.0)
        
        # Calculate cosmological quantities
        scale_factor = 1.0 / (1.0 + z) if z >= 0 else 1.0
        age = self._cosmology.age_of_universe(z)
        comoving_dist = self._cosmology.comoving_distance(z) if z > 0 else 0.0

        self.update_metadata(
            scale_factor=scale_factor,
            age_universe_gyr=age,
            comoving_distance_mpc=comoving_dist,
            hubble_param=self._cosmology.hubble_parameter(z),
        )

    @property
    def positions(self) -> torch.Tensor:
        """Get particle positions [N, 3]."""
        return self._data

    @property
    def features(self) -> Optional[torch.Tensor]:
        """Get particle features [N, D]."""
        return self._features

    @property
    def edge_index(self) -> Optional[torch.Tensor]:
        """Get graph edge indices [2, E]."""
        return self._edge_index

    @property
    def num_particles(self) -> int:
        """Number of particles."""
        return self._data.shape[0]

    @property
    def simulation_name(self) -> str:
        """Simulation name."""
        return self.get_metadata("simulation_name", "unknown")

    @property
    def particle_type(self) -> str:
        """Particle type."""
        return self.get_metadata("particle_type", "unknown")

    @property
    def box_size(self) -> float:
        """Simulation box size."""
        return self.get_metadata("box_size", 0.0)

    @property
    def redshift(self) -> float:
        """Simulation redshift."""
        return self.get_metadata("redshift", 0.0)

    @property
    def cosmology(self) -> "CosmologyCalculator":
        """Cosmology calculator."""
        return self._cosmology

    def periodic_distance(
        self, 
        pos1: Union[torch.Tensor, int], 
        pos2: Union[torch.Tensor, int]
    ) -> torch.Tensor:
        """
        Calculate distance with periodic boundary conditions.
        
        Args:
            pos1: Position tensor [3] or [N, 3], or particle index
            pos2: Position tensor [3] or [N, 3], or particle index
            
        Returns:
            Minimum distance accounting for periodic boundaries
        """
        # Handle particle indices
        if isinstance(pos1, int):
            pos1 = self.positions[pos1]
        if isinstance(pos2, int):
            pos2 = self.positions[pos2]

        pos1 = torch.as_tensor(pos1)
        pos2 = torch.as_tensor(pos2)

        # Calculate difference
        diff = pos2 - pos1

        # Apply periodic boundary conditions
        box_size = self.box_size
        diff = diff - box_size * torch.round(diff / box_size)

        # Calculate distance
        if diff.dim() == 1:
            return torch.linalg.norm(diff)
        else:
            return torch.linalg.norm(diff, dim=-1)

    def apply_periodic_boundaries(self, positions: torch.Tensor) -> torch.Tensor:
        """Apply periodic boundary conditions to positions."""
        box_size = self.box_size
        # Proper modulo operation that handles negative values
        return positions - box_size * torch.floor(positions / box_size)

    def get_particle_subset(self, mask: torch.Tensor) -> "SimulationTensor":
        """
        Get subset of particles based on boolean mask.
        
        Args:
            mask: Boolean mask [N]
            
        Returns:
            New SimulationTensor with selected particles
        """
        new_positions = self.positions[mask]
        new_features = self.features[mask] if self.features is not None else None
        
        # Filter edge_index if present (more complex)
        new_edge_index = None
        if self.edge_index is not None:
            # This would need proper graph filtering - simplified for now
            logger.warning("Edge filtering not implemented - edges will be removed")

        # Extract metadata without tensor_type to avoid conflicts
        subset_metadata = {k: v for k, v in self._metadata.items() if k != 'tensor_type'}
        
        return SimulationTensor(
            positions=new_positions,
            features=new_features,
            edge_index=new_edge_index,
            **subset_metadata
        )

    def calculate_center_of_mass(self) -> torch.Tensor:
        """Calculate center of mass of particles."""
        if self.features is not None and self.features.shape[1] > 0:
            # Assume first feature is mass
            masses = self.features[:, 0]
            com = torch.sum(self.positions * masses.unsqueeze(-1), dim=0) / torch.sum(masses)
        else:
            # Equal mass assumption
            com = torch.mean(self.positions, dim=0)
        
        return com

    def to_torch_geometric(self) -> "torch_geometric.data.Data":
        """
        Convert to PyTorch Geometric Data object.
        
        Returns:
            PyTorch Geometric Data object
        """
        try:
            import torch_geometric
        except ImportError:
            raise ImportError("PyTorch Geometric required for graph conversion")

        data = torch_geometric.data.Data(
            pos=self.positions,
            x=self.features,
            edge_index=self.edge_index,
        )
        
        # Add metadata as attributes
        for key, value in self._metadata.items():
            if isinstance(value, (int, float, str, bool)):
                setattr(data, key, value)

        return data

    @classmethod
    def from_torch_geometric(cls, data: "torch_geometric.data.Data", **kwargs) -> "SimulationTensor":
        """
        Create SimulationTensor from PyTorch Geometric Data object.
        
        Args:
            data: PyTorch Geometric Data object
            **kwargs: Additional metadata
            
        Returns:
            New SimulationTensor
        """
        # Extract core data
        positions = data.pos
        features = getattr(data, 'x', None)
        edge_index = getattr(data, 'edge_index', None)

        # Extract metadata from data attributes
        metadata = {}
        for attr in ['simulation_name', 'particle_type', 'box_size', 'redshift']:
            if hasattr(data, attr):
                metadata[attr] = getattr(data, attr)

        metadata.update(kwargs)

        return cls(
            positions=positions,
            features=features,
            edge_index=edge_index,
            **metadata
        )

    def update_redshift(self, new_redshift: float) -> None:
        """Update redshift and recalculate cosmological quantities."""
        self.update_metadata(redshift=new_redshift)
        self._update_cosmological_metadata()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SimulationTensor({self.simulation_name} {self.particle_type}: "
            f"{self.num_particles} particles, z={self.redshift:.3f}, "
            f"box={self.box_size:.1f})"
        )


class CosmologyCalculator:
    """
    Lightweight cosmology calculator integrated with SimulationTensor.
    
    Provides cosmological distance calculations, age of universe,
    and Hubble parameter for simulation analysis.
    """

    def __init__(
        self,
        H0: float = 70.0,
        Omega_m: float = 0.3,
        Omega_Lambda: float = 0.7,
        Omega_k: float = 0.0,
    ):
        """Initialize cosmology calculator."""
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda
        self.Omega_k = Omega_k

        # Validate parameters
        total_omega = Omega_m + Omega_Lambda + Omega_k
        if abs(total_omega - 1.0) > 1e-6:
            logger.warning(f"Omega parameters don't sum to 1: {total_omega:.6f}")

    def hubble_parameter(self, z: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Calculate Hubble parameter H(z)."""
        if isinstance(z, torch.Tensor):
            E_z = torch.sqrt(
                self.Omega_m * (1 + z)**3 
                + self.Omega_k * (1 + z)**2 
                + self.Omega_Lambda
            )
            return self.H0 * E_z
        else:
            z = np.asarray(z)
            E_z = np.sqrt(
                self.Omega_m * (1 + z)**3 
                + self.Omega_k * (1 + z)**2 
                + self.Omega_Lambda
            )
            return self.H0 * E_z

    def comoving_distance(self, z: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Calculate comoving distance to redshift z."""
        if SCIPY_AVAILABLE:
            return self._comoving_distance_scipy(z)
        else:
            return self._comoving_distance_fallback(z)

    def _comoving_distance_scipy(self, z):
        """Calculate comoving distance using scipy integration."""
        c_km_s = 299792.458  # Speed of light in km/s
        
        def integrand(z_prime):
            return 1.0 / self.hubble_parameter(z_prime) * self.H0
        
        if np.isscalar(z):
            result, _ = scipy.integrate.quad(integrand, 0, z)
            return c_km_s * result
        else:
            results = []
            z_array = np.asarray(z)
            for z_val in z_array.flat:
                result, _ = scipy.integrate.quad(integrand, 0, z_val)
                results.append(c_km_s * result)
            return np.array(results).reshape(z_array.shape)

    def _comoving_distance_fallback(self, z):
        """Calculate comoving distance using simple trapezoid integration."""
        c_km_s = 299792.458  # Speed of light in km/s
        
        z = np.asarray(z)
        scalar_input = z.ndim == 0
        z = np.atleast_1d(z)
        
        results = []
        for z_val in z:
            if z_val <= 0:
                results.append(0.0)
                continue
                
            # Simple trapezoid rule integration
            n_steps = max(100, int(z_val * 50))  # Adaptive step size
            z_grid = np.linspace(0, z_val, n_steps + 1)
            integrand = 1.0 / (self.hubble_parameter(z_grid) / self.H0)
            
            # Trapezoid rule
            dz = z_grid[1] - z_grid[0]
            integral = dz * (0.5 * integrand[0] + np.sum(integrand[1:-1]) + 0.5 * integrand[-1])
            
            results.append(c_km_s * integral)
        
        results = np.array(results)
        return results[0] if scalar_input else results

    def angular_diameter_distance(self, z: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Calculate angular diameter distance."""
        D_c = self.comoving_distance(z)
        return D_c / (1 + z)

    def luminosity_distance(self, z: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Calculate luminosity distance."""
        D_c = self.comoving_distance(z)
        return D_c * (1 + z)

    def age_of_universe(self, z: Union[float, np.ndarray, torch.Tensor] = 0) -> Union[float, np.ndarray, torch.Tensor]:
        """Calculate age of universe at redshift z."""
        if SCIPY_AVAILABLE:
            return self._age_scipy(z)
        else:
            return self._age_fallback(z)

    def _age_scipy(self, z):
        """Calculate age using scipy integration."""
        H0_s = self.H0 / 3.086e19  # Convert to 1/s
        
        def integrand(z_prime):
            return 1.0 / ((1 + z_prime) * self.hubble_parameter(z_prime) / self.H0)
        
        if np.isscalar(z):
            result, _ = scipy.integrate.quad(integrand, z, np.inf)
            return result / H0_s / (365.25 * 24 * 3600 * 1e9)  # Convert to Gyr
        else:
            results = []
            for z_val in np.asarray(z).flat:
                result, _ = scipy.integrate.quad(integrand, z_val, np.inf)
                results.append(result / H0_s / (365.25 * 24 * 3600 * 1e9))
            return np.array(results).reshape(np.asarray(z).shape)

    def _age_fallback(self, z):
        """Calculate age using simple approximation."""
        # Simple approximation: t ≈ 2/(3*H0) in flat universe
        H0_gyr = self.H0 / 97.8  # Convert to 1/Gyr
        return 2.0 / (3.0 * H0_gyr * np.sqrt(self.Omega_m) * (1 + z)**(3/2))


__all__ = [
    "SimulationTensor",
    "CosmologyCalculator",
] 