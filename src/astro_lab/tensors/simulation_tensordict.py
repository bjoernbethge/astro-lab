"""
TensorDict-basierte Implementierung für Simulationsdaten
======================================================

Umstellung der SimulationTensor und verwandter Klassen auf TensorDict-Architektur.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict

from .tensordict_astro import AstroTensorDict, SpatialTensorDict


class SimulationTensorDict(AstroTensorDict):
    """
    TensorDict for N-Body simulations and cosmological data.

    Structure:
    {
        "positions": Tensor[N, 3],    # Particle positions
        "velocities": Tensor[N, 3],   # Particle velocities
        "masses": Tensor[N],          # Particle masses
        "potential": Tensor[N],       # Gravitational potential
        "forces": Tensor[N, 3],       # Forces (optional)
        "meta": {
            "simulation_type": str,
            "time_step": float,
            "current_time": float,
            "units": Dict[str, str],
            "cosmology": Dict[str, float],
        }
    }
    """

    def __init__(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        masses: torch.Tensor,
        simulation_type: str = "nbody",
        time_step: float = 0.01,
        current_time: float = 0.0,
        cosmology: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Initialize SimulationTensorDict.

        Args:
            positions: [N, 3] Particle positions
            velocities: [N, 3] Particle velocities
            masses: [N] Particle masses
            simulation_type: Type of simulation
            time_step: Time step
            current_time: Current time
            cosmology: Cosmological parameters
            **kwargs: Additional arguments
        """
        if positions.shape != velocities.shape:
            raise ValueError("Positions and velocities must have same shape")

        if positions.shape[0] != masses.shape[0]:
            raise ValueError("Number of positions must match number of masses")

        n_objects = positions.shape[0]

        # Default cosmology (Planck 2018)
        if cosmology is None:
            cosmology = {
                "H0": 67.4,  # Hubble constant
                "Omega_m": 0.315,  # Matter density
                "Omega_L": 0.685,  # Dark energy density
                "Omega_b": 0.049,  # Baryon density
            }

        data = {
            "positions": positions,
            "velocities": velocities,
            "masses": masses,
            "potential": torch.zeros(n_objects),
            "forces": torch.zeros_like(positions),
            "meta": {
                "simulation_type": simulation_type,
                "time_step": time_step,
                "current_time": current_time,
                "units": {
                    "length": "kpc",
                    "velocity": "km/s",
                    "mass": "solar_mass",
                    "time": "Myr",
                },
                "cosmology": cosmology,
            },
        }

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    @property
    def positions(self) -> torch.Tensor:
        """Particle positions."""
        return self["positions"]

    @property
    def velocities(self) -> torch.Tensor:
        """Particle velocities."""
        return self["velocities"]

    @property
    def masses(self) -> torch.Tensor:
        """Particle masses."""
        return self["masses"]

    def compute_gravitational_forces(self, softening: float = 0.1) -> torch.Tensor:
        """
        Computes gravitational forces between all particles.

        Args:
            softening: Softening parameter to avoid singularities

        Returns:
            [N, 3] Force tensor
        """
        positions = self.positions
        masses = self.masses
        n_particles = positions.shape[0]

        # Pairwise distances
        r_ij = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 3]
        distances = torch.norm(r_ij, dim=-1)  # [N, N]

        # Apply softening
        distances = torch.sqrt(distances**2 + softening**2)

        # Gravitational forces: F = G * m1 * m2 / r^2 * r_hat
        G = 4.301e-6  # Gravitational constant in kpc * (km/s)^2 / solar_mass

        # Avoid self-interaction
        mask = torch.eye(n_particles, dtype=torch.bool, device=positions.device)
        distances = distances.masked_fill(mask, float("inf"))

        # Force magnitudes
        force_magnitudes = G * masses.unsqueeze(1) * masses.unsqueeze(0) / distances**2
        force_magnitudes = force_magnitudes.masked_fill(mask, 0)

        # Force directions
        force_directions = r_ij / distances.unsqueeze(-1)
        force_directions = force_directions.masked_fill(mask.unsqueeze(-1), 0)

        # Total forces
        forces = -torch.sum(force_magnitudes.unsqueeze(-1) * force_directions, dim=1)

        self["forces"] = forces
        return forces

    def compute_potential_energy(self, softening: float = 0.1) -> torch.Tensor:
        """
        Computes gravitational potential for each particle.

        Args:
            softening: Softening parameter

        Returns:
            [N] Potential tensor
        """
        positions = self.positions
        masses = self.masses
        n_particles = positions.shape[0]

        # Pairwise distances
        r_ij = positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.norm(r_ij, dim=-1)

        # Apply softening
        distances = torch.sqrt(distances**2 + softening**2)

        # Avoid self-interaction
        mask = torch.eye(n_particles, dtype=torch.bool, device=positions.device)
        distances = distances.masked_fill(mask, float("inf"))

        G = 4.301e-6  # Gravitational constant

        # Potential: Phi = -G * sum(m_j / r_ij)
        potential = -G * torch.sum(masses.unsqueeze(0) / distances, dim=1)

        self["potential"] = potential
        return potential

    def leapfrog_step(self) -> SimulationTensorDict:
        """
        Performs a Leapfrog integration step.

        Returns:
            New SimulationTensorDict with updated state
        """
        dt = self["meta", "time_step"]

        # Compute current forces
        forces = self.compute_gravitational_forces()
        accelerations = forces / self.masses.unsqueeze(-1)

        # Leapfrog integration
        # v(t + dt/2) = v(t) + a(t) * dt/2
        half_step_velocities = self.velocities + accelerations * dt / 2

        # x(t + dt) = x(t) + v(t + dt/2) * dt
        new_positions = self.positions + half_step_velocities * dt

        # Create new state for force computation
        temp_sim = SimulationTensorDict(
            positions=new_positions,
            velocities=half_step_velocities,
            masses=self.masses,
            simulation_type=self["meta", "simulation_type"],
            time_step=dt,
            current_time=self["meta", "current_time"],
            cosmology=self["meta", "cosmology"],
        )

        # Compute new forces
        new_forces = temp_sim.compute_gravitational_forces()
        new_accelerations = new_forces / self.masses.unsqueeze(-1)

        # v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        new_velocities = half_step_velocities + new_accelerations * dt / 2

        return SimulationTensorDict(
            positions=new_positions,
            velocities=new_velocities,
            masses=self.masses,
            simulation_type=self["meta", "simulation_type"],
            time_step=dt,
            current_time=self["meta", "current_time"] + dt,
            cosmology=self["meta", "cosmology"],
        )

    def compute_kinetic_energy(self) -> torch.Tensor:
        """Computes kinetic energy."""
        velocities = self.velocities
        masses = self.masses

        # E_kin = 1/2 * m * v^2
        kinetic_energy = 0.5 * masses * torch.sum(velocities**2, dim=-1)
        return kinetic_energy

    def compute_total_energy(self) -> torch.Tensor:
        """Computes total energy of the system."""
        kinetic = torch.sum(self.compute_kinetic_energy())
        potential = torch.sum(self.compute_potential_energy())
        return kinetic + potential

    def compute_center_of_mass(self) -> torch.Tensor:
        """Computes center of mass of the system."""
        total_mass = torch.sum(self.masses)
        com = torch.sum(self.positions * self.masses.unsqueeze(-1), dim=0) / total_mass
        return com

    def compute_angular_momentum(self) -> torch.Tensor:
        """Computes total angular momentum of the system."""
        com = self.compute_center_of_mass()
        relative_positions = self.positions - com

        # L = sum(m_i * r_i x v_i)
        angular_momentum = torch.sum(
            self.masses.unsqueeze(-1)
            * torch.cross(relative_positions, self.velocities),
            dim=0,
        )
        return angular_momentum

    def to_spatial_tensor(self) -> SpatialTensorDict:
        """Converts to SpatialTensorDict."""
        return SpatialTensorDict(
            coordinates=self.positions,
            coordinate_system="cartesian",
            unit=self["meta", "units", "length"],
        )

    def run_simulation(
        self, n_steps: int, save_interval: int = 1
    ) -> List[SimulationTensorDict]:
        """
        Runs N-Body simulation over multiple time steps.

        Args:
            n_steps: Number of time steps
            save_interval: Save interval

        Returns:
            List of SimulationTensorDict snapshots
        """
        snapshots = []
        current_sim = self

        for step in range(n_steps):
            if step % save_interval == 0:
                snapshots.append(current_sim)

            current_sim = current_sim.leapfrog_step()

            # Optional: Energy conservation check
            if step % 100 == 0:
                energy = current_sim.compute_total_energy()
                print(f"Step {step}: Total Energy = {energy:.6f}")

        # Last state addition
        snapshots.append(current_sim)

        return snapshots


class CosmologyTensorDict(AstroTensorDict):
    """
    TensorDict for cosmological calculations.

    Structure:
    {
        "redshifts": Tensor[N],       # Redshifts
        "distances": TensorDict,      # Various distance measures
        "times": TensorDict,          # Time measures
        "meta": {
            "cosmology": Dict[str, float],
            "calculated_quantities": List[str],
        }
    }
    """

    def __init__(
        self,
        redshifts: torch.Tensor,
        cosmology: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Initialises CosmologyTensorDict.

        Args:
            redshifts: [N] Redshifts
            cosmology: Cosmological parameters
        """
        if cosmology is None:
            cosmology = {
                "H0": 67.4,  # km/s/Mpc
                "Omega_m": 0.315,
                "Omega_L": 0.685,
                "Omega_k": 0.0,
            }

        n_objects = redshifts.shape[0]

        data = {
            "redshifts": redshifts,
            "distances": TensorDict({}, batch_size=(n_objects,)),
            "times": TensorDict({}, batch_size=(n_objects,)),
            "meta": {
                "cosmology": cosmology,
                "calculated_quantities": [],
            },
        }

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    def compute_luminosity_distance(self) -> torch.Tensor:
        """Computes luminosity distance."""
        z = self["redshifts"]
        cosmo = self["meta", "cosmology"]

        # Simplified calculation for flat universe
        H0 = cosmo["H0"]
        Om = cosmo["Omega_m"]
        OL = cosmo["Omega_L"]

        # Mobile distance (simplified)
        c = 299792.458  # km/s

        # Integral approximation
        a_values = torch.linspace(1 / (1 + z.max()), 1, 1000)
        integrand = 1 / torch.sqrt(Om * (1 / a_values) ** 3 + OL)
        comoving_distance = c / H0 * torch.trapz(integrand, a_values)

        # Luminosity distance
        luminosity_distance = (1 + z) * comoving_distance

        self["distances", "luminosity"] = luminosity_distance
        self["meta", "calculated_quantities"].append("luminosity_distance")

        return luminosity_distance

    def compute_age_at_redshift(self) -> torch.Tensor:
        """Computes age of the universe at given redshift."""
        z = self["redshifts"]
        cosmo = self["meta", "cosmology"]

        H0 = cosmo["H0"]
        Om = cosmo["Omega_m"]
        OL = cosmo["Omega_L"]

        # Hubble time
        t_H = 9.78e9 / H0  # Gyr

        # Simplified age calculation
        age = (
            t_H
            * torch.sqrt(OL / Om)
            * torch.log(
                (1 + torch.sqrt(OL / Om))
                / (torch.sqrt(OL / Om) + torch.sqrt(OL / Om + (1 + z) ** 3))
            )
        )

        self["times", "age"] = age
        self["meta", "calculated_quantities"].append("age_at_redshift")

        return age
