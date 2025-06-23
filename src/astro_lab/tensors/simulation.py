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
from pydantic import Field, field_validator
from torch_geometric.data import Data as PyGData

from .base import AstroTensorBase

logger = logging.getLogger(__name__)


class SimulationTensor(AstroTensorBase):
    """
    Tensor for cosmological simulation data with integrated cosmology.

    The `data` tensor is expected to contain at least 3 columns for spatial
    positions (x, y, z). Additional columns can represent features like
    mass, velocity, metallicity, etc.

    Handles:
    - Particle positions, velocities, masses, and other features.
    - Cosmological calculations (redshift, distances, age).
    - Simulation-specific metadata (box size, particle types).
    - Periodic boundary conditions.
    - Graph structure for spatial relationships.
    """

    data: torch.Tensor
    simulation_name: str = Field("TNG50", description="e.g., 'TNG50', 'Illustris'")
    particle_type: str = Field("gas", description="'gas', 'dark_matter', 'stars', 'black_holes'")
    box_size: float = Field(35000.0, description="Simulation box size, e.g., in ckpc/h")
    redshift: float = Field(0.0, description="Simulation redshift")
    cosmology: Dict[str, float] = Field(
        default_factory=lambda: {"H0": 70.0, "Omega_m": 0.3, "Omega_Lambda": 0.7},
        description="Cosmological parameters",
    )
    edge_index: Optional[torch.Tensor] = Field(None, description="Graph edge indices [2, E]")
    feature_names: List[str] = Field(
        default_factory=list, description="Names of feature columns beyond positions"
    )

    @field_validator("data")
    def validate_simulation_data(cls, v):
        if v.ndim != 2 or v.shape[1] < 3:
            raise ValueError(f"Simulation data must be a 2D tensor with at least 3 columns for positions, but got shape {v.shape}")
        return v

    @field_validator("edge_index")
    def validate_edge_index(cls, v):
        if v is not None and (v.ndim != 2 or v.shape[0] != 2):
            raise ValueError(f"edge_index must be a 2D tensor with shape [2, E], but got shape {v.shape}")
        return v

    def __init__(self, data: torch.Tensor, **kwargs):
        super().__init__(data=data, **kwargs)
        self._cosmology_calculator = CosmologyCalculator(**self.cosmology)
        self._update_cosmological_metadata()

    def _update_cosmological_metadata(self) -> None:
        """Update cosmological metadata based on redshift."""
        z = self.redshift
        calc = self._cosmology_calculator
        scale_factor = 1.0 / (1.0 + z) if z >= 0 else 1.0
        age = calc.age_of_universe(z)
        comoving_dist = calc.comoving_distance(z) if z > 0 else 0.0
        hubble = calc.hubble_parameter(z)

        self.meta.update({
            "scale_factor": scale_factor,
            "age_universe_gyr": age,
            "comoving_distance_mpc": comoving_dist,
            "hubble_param": hubble,
            "snap_num": self.meta.get("snap_num"),
            "time": self.meta.get("time"),
            "mass_table": self.meta.get("mass_table"),
            "periodic_boundaries": self.meta.get("periodic_boundaries", True),
        })


    @property
    def positions(self) -> torch.Tensor:
        """Get particle positions [N, 3]."""
        return self.data[:, :3]

    @property
    def features(self) -> Optional[torch.Tensor]:
        """Get particle features [N, D] if they exist."""
        if self.data.shape[1] > 3:
            return self.data[:, 3:]
        return None

    @property
    def num_particles(self) -> int:
        """Number of particles."""
        return self.data.shape[0]

    def periodic_distance(self, idx1: int, idx2: int) -> float:
        """Calculate the minimum distance between two particles, accounting for periodic boundaries."""
        pos1 = self.positions[idx1]
        pos2 = self.positions[idx2]
        diff = pos1 - pos2
        diff = diff - self.box_size * torch.round(diff / self.box_size)
        return torch.linalg.norm(diff).item()

    def apply_periodic_boundaries(self, positions: torch.Tensor) -> torch.Tensor:
        """Apply periodic boundary conditions to a set of positions."""
        return positions - self.box_size * torch.floor(positions / self.box_size)

    def get_particle_subset(self, mask: torch.Tensor) -> "SimulationTensor":
        """Get a subset of particles based on a boolean mask."""
        if mask.dtype != torch.bool or mask.ndim != 1 or mask.shape[0] != self.num_particles:
            raise ValueError("Mask must be a 1D boolean tensor of length `num_particles`")

        new_data = self.data[mask]
        
        # Note: Edge index filtering is complex and not handled here for simplicity.
        # A robust implementation would require re-indexing.
        if self.edge_index is not None:
            logger.warning("Edge index is not filtered in subset. Creating new tensor without edges.")

        return self.__class__(
            data=new_data,
            **self.model_dump(exclude={"data", "edge_index"})
        )

    def calculate_center_of_mass(self, mass_feature_idx: Optional[int] = None) -> torch.Tensor:
        """Calculate the center of mass."""
        if self.features is None and mass_feature_idx is not None:
            raise ValueError("Cannot calculate center of mass by feature index when no features are present.")

        if mass_feature_idx is not None:
            if mass_feature_idx >= self.features.shape[1]:
                raise IndexError("mass_feature_idx is out of bounds for features tensor.")
            masses = self.features[:, mass_feature_idx]
        else:
            # Assume equal masses if not provided
            masses = torch.ones(self.num_particles, device=self.device)

        total_mass = torch.sum(masses)
        if total_mass == 0:
            return torch.zeros(3, device=self.device)

        com = torch.sum(self.positions * masses.unsqueeze(1), dim=0) / total_mass
        return self.apply_periodic_boundaries(com)

    def to_torch_geometric(self) -> PyGData:
        """Convert to a torch_geometric.data.Data object."""
        return PyGData(
            x=self.features,
            pos=self.positions,
            edge_index=self.edge_index,
            **self.meta,
        )

    @classmethod
    def from_torch_geometric(cls, data: PyGData, **kwargs) -> "SimulationTensor":
        """Create a SimulationTensor from a torch_geometric.data.Data object."""
        if not hasattr(data, 'pos'):
            raise ValueError("PyGData object must have a 'pos' attribute for positions.")
        
        tensor_data = data.pos
        if hasattr(data, 'x') and data.x is not None:
            tensor_data = torch.cat([data.pos, data.x], dim=1)

        # Collect metadata from PyGData object and kwargs
        sim_kwargs = {k: v for k, v in data if k not in ['x', 'pos', 'edge_index']}
        sim_kwargs.update(kwargs)
        
        return cls(
            data=tensor_data,
            edge_index=data.edge_index,
            **sim_kwargs
        )

    def update_redshift(self, new_redshift: float):
        """Update the redshift and recalculate dependent cosmological properties."""
        self.redshift = new_redshift
        self._update_cosmological_metadata()

    def to_pyvista(self, scalars: Optional[Union[str, torch.Tensor]] = None) -> Any:
        """Convert particle positions to a PyVista PolyData object for 3D visualization."""
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("PyVista is required for this functionality. `pip install pyvista`")
        
        points = self.positions.detach().cpu().numpy()
        poly = pv.PolyData(points)

        if scalars is not None:
            if isinstance(scalars, str):
                if self.features is None or scalars not in self.feature_names:
                    raise ValueError(f"Feature '{scalars}' not found in feature_names.")
                scalar_data = self.features[:, self.feature_names.index(scalars)]
            else:
                scalar_data = torch.as_tensor(scalars)

            poly[str(scalars) if isinstance(scalars, str) else "scalars"] = scalar_data.detach().cpu().numpy()

        return poly

    def to_blender(self, name: str = "simulation") -> Dict[str, Any]:
        """Generate a dictionary compatible with the Blender live tensor bridge."""
        return {
            "name": name,
            "type": "geometry",
            "positions": self.positions.detach().cpu().numpy(),
            "edges": self.edge_index.detach().cpu().numpy().T if self.edge_index is not None else [],
            "metadata": self.model_dump_json()
        }
        
    def memory_info(self) -> Dict[str, str]:
        """Returns information about the tensor's memory usage."""
        total_bytes = self.data.element_size() * self.data.nelement()
        if self.edge_index is not None:
            total_bytes += self.edge_index.element_size() * self.edge_index.nelement()
        
        gb = total_bytes / (1024 ** 3)
        return {"total_size_gb": f"{gb:.4f}"}

    def __repr__(self) -> str:
        base_repr = super().__repr__()
        return f"{base_repr[:-1]}, sim='{self.simulation_name}', particles={self.num_particles}, z={self.redshift:.2f})"

# Note: CosmologyCalculator remains largely the same as it's a helper.
# It does not depend on the tensor structure, only on cosmological parameters.
class CosmologyCalculator:
    """
    A helper class to perform cosmological calculations.
    It can be initialized with different cosmological parameters.
    """
    def __init__(
        self,
        H0: float = 70.0,
        Omega_m: float = 0.3,
        Omega_Lambda: float = 0.7,
        T_cmb: float = 2.725,
    ):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda
        self.Omega_k = 1.0 - Omega_m - Omega_Lambda
        self.T_cmb = T_cmb

        # Constants
        self.c = 299792.458  # Speed of light in km/s
        self.H0_per_s = H0 / (3.086e19)  # H0 in 1/s
        self.yr_to_s = 3.154e7

    def hubble_parameter(self, z: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Hubble parameter H(z) in km/s/Mpc."""
        z_tensor = torch.as_tensor(z, dtype=torch.float32)
        E_z = torch.sqrt(
            self.Omega_m * (1 + z_tensor)**3 +
            self.Omega_k * (1 + z_tensor)**2 +
            self.Omega_Lambda
        )
        return self.H0 * E_z

    def _comoving_distance_integrand(self, z_prime):
        return 1.0 / self.hubble_parameter(z_prime)

    def comoving_distance(self, z: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Comoving distance in Mpc."""
        is_tensor = isinstance(z, torch.Tensor)
        input_val = z.detach().cpu().numpy() if is_tensor else z

        def single_comoving_distance(z_val):
            if z_val <= 0:
                return 0.0
            # Scipy can be slow for single values, but it's robust.
            # For performance-critical code, a custom integrator or approximation might be better.
            from scipy.integrate import quad
            dist, _ = quad(self._comoving_distance_integrand, 0, z_val)
            return self.c * dist

        vectorized_dist = np.vectorize(single_comoving_distance)
        result = vectorized_dist(input_val)

        return torch.from_numpy(result).float() if is_tensor else result

    def angular_diameter_distance(self, z: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Angular diameter distance in Mpc."""
        dc = self.comoving_distance(z)
        return dc / (1 + torch.as_tensor(z, dtype=torch.float32))

    def luminosity_distance(self, z: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Luminosity distance in Mpc."""
        dc = self.comoving_distance(z)
        return dc * (1 + torch.as_tensor(z, dtype=torch.float32))

    def _age_integrand(self, z):
        return 1.0 / ((1 + z) * self.hubble_parameter(z))

    def age_of_universe(self, z: Union[float, torch.Tensor] = 0) -> Union[float, torch.Tensor]:
        """Age of the universe at a given redshift in Gyr."""
        # This integral is from z to infinity
        z_max = 1000 # Approximation for infinity
        z_steps = torch.logspace(torch.log10(torch.as_tensor(z, dtype=torch.float32) + 1e-6), torch.log10(torch.as_tensor(z_max)), 1000)
        integrand_values = self._age_integrand(z_steps)
        t0 = 978.46 / self.H0 # Age of universe factor
        return t0 * torch.trapz(integrand_values, z_steps)


__all__ = [
    "SimulationTensor",
    "CosmologyCalculator",
]
