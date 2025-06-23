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
from pydantic import Field, field_validator, model_validator, PrivateAttr
from typing_extensions import Self
from torch_geometric.data import Data as PyGData

from .base import AstroTensorBase

logger = logging.getLogger(__name__)


# Note: CosmologyCalculator is a helper and does not need to be a Pydantic model.
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
        self.T_cmb = T_cmb
        self.h = H0 / 100.0
        self.Omega_k = 1.0 - Omega_m - Omega_Lambda

    def hubble_parameter(self, z: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Calculate Hubble parameter H(z)."""
        return self.H0 * torch.sqrt(
            self.Omega_m * (1 + z) ** 3
            + self.Omega_k * (1 + z) ** 2
            + self.Omega_Lambda
        )

    def _comoving_distance_integrand(self, z_prime):
        return 1.0 / self.hubble_parameter(z_prime)

    def comoving_distance(self, z: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Calculate comoving distance (in Mpc)."""
        from scipy.integrate import quad
        
        def single_comoving_distance(z_val):
            c = 299792.458  # Speed of light in km/s
            distance, _ = quad(self._comoving_distance_integrand, 0, z_val)
            return c * distance

        if isinstance(z, (np.ndarray, torch.Tensor)):
            return np.vectorize(single_comoving_distance)(z)
        return single_comoving_distance(z)

    def angular_diameter_distance(self, z: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Calculate angular diameter distance."""
        return self.comoving_distance(z) / (1 + z)

    def luminosity_distance(self, z: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Calculate luminosity distance."""
        return self.comoving_distance(z) * (1 + z)

    def _age_integrand(self, z):
        return 1.0 / ((1 + z) * self.hubble_parameter(z))

    def age_of_universe(self, z: Union[float, torch.Tensor] = 0) -> Union[float, torch.Tensor]:
        """Calculate the age of the universe at redshift z (in Gyr)."""
        from scipy.integrate import quad
        
        # FIX: Handle tensor input for logspace
        if isinstance(z, torch.Tensor) and z.dim() > 0:
            ages = [self.age_of_universe(zi.item()) for zi in z]
            return torch.tensor(ages, device=z.device, dtype=z.dtype)

        # Integration from z to infinity
        integral, _ = quad(self._age_integrand, z, np.inf)
        
        # Conversion factor from km/s/Mpc to 1/Gyr
        km_per_mpc = 3.0857e19
        s_per_gyr = 3.1536e16
        conv_factor = km_per_mpc / s_per_gyr
        
        return integral * conv_factor


class SimulationTensor(AstroTensorBase):
    """
    Tensor for cosmological simulation data with integrated cosmology.
    """
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
    _cosmology_calculator: CosmologyCalculator = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook to set up the cosmology calculator."""
        super().model_post_init(__context)
        self._cosmology_calculator = CosmologyCalculator(**self.cosmology)
        self._update_cosmological_metadata()

    @field_validator("data")
    def validate_simulation_data(cls, v):
        if v.ndim != 2 or v.shape[1] < 3:
            raise ValueError(f"Simulation data must be a 2D tensor with at least 3 columns for positions, but got shape {v.shape}")
        return v
    
    def _update_cosmological_metadata(self) -> None:
        """Update cosmological metadata based on redshift."""
        z = self.redshift
        calc = self._cosmology_calculator
        scale_factor = 1.0 / (1.0 + z) if z >= 0 else 1.0
        age = calc.age_of_universe(z)
        comoving_dist = calc.comoving_distance(z) if z > 0 else 0.0
        hubble = calc.hubble_parameter(z)

        # Pydantic models are immutable, so we update through a new model instance
        # This is a bit tricky since we're in post-init. A better pattern would be
        # to have these as computed fields if they need to be part of the model state.
        # For now, we'll store them in the generic `meta` dict.
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
        
        if self.edge_index is not None:
            logger.warning("Edge index is not filtered in subset. Creating new tensor without edges.")
        
        # Use the robust _create_new_instance helper from the base class
        return self._create_new_instance(new_data=new_data, edge_index=None)

    def calculate_center_of_mass(self, mass_feature_idx: Optional[int] = None) -> torch.Tensor:
        """Calculate the center of mass."""
        if self.features is None and mass_feature_idx is not None:
            raise ValueError("Cannot calculate center of mass by feature index when no features are present.")

        if mass_feature_idx is not None:
            if mass_feature_idx >= self.features.shape[1]:
                raise IndexError("mass_feature_idx is out of bounds for features tensor.")
            masses = self.features[:, mass_feature_idx]
        else:
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
            simulation_name=self.simulation_name,
            particle_type=self.particle_type,
            box_size=self.box_size,
            redshift=self.redshift
        )

    @classmethod
    def from_torch_geometric(cls, data: PyGData, **kwargs) -> "SimulationTensor":
        """Create a SimulationTensor from a torch_geometric.data.Data object."""
        if not hasattr(data, 'pos'):
            raise ValueError("PyGData object must have a 'pos' attribute for positions.")
        
        tensor_data = data.pos
        if hasattr(data, 'x') and data.x is not None:
            tensor_data = torch.cat([data.pos, data.x], dim=1)

        sim_kwargs = {k: v for k, v in data if k not in ['x', 'pos', 'edge_index']}
        sim_kwargs.update(kwargs)
        
        return cls(
            data=tensor_data,
            edge_index=data.edge_index,
            **sim_kwargs
        )

    def update_redshift(self, new_redshift: float):
        """Update the redshift and recalculate dependent cosmological properties."""
        # Pydantic models are immutable, this operation should return a new instance
        # For now, we mutate in place, which is not ideal.
        object.__setattr__(self, 'redshift', new_redshift)
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
        # FIX: Provide a default function to handle tensor serialization
        def tensor_serializer(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        return {
            "name": name,
            "type": "geometry",
            "positions": self.positions.detach().cpu().numpy(),
            "edges": self.edge_index.detach().cpu().numpy().T if self.edge_index is not None else [],
            "metadata": self.model_dump_json(exclude={'data', 'edge_index'}, default=tensor_serializer)
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


__all__ = [
    "SimulationTensor",
    "CosmologyCalculator",
]
