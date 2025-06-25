"""
TensorDict-basierte Implementierung für Simulationsdaten
======================================================

Umstellung der SimulationTensor und verwandter Klassen auf TensorDict-Architektur.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tensordict import TensorDict
import numpy as np

from .tensordict_astro import AstroTensorDict
from .spatial_tensordict import SpatialTensorDict


class SimulationTensorDict(AstroTensorDict):
    """
    TensorDict für N-Body-Simulationen und kosmologische Daten.

    Struktur:
    {
        "positions": Tensor[N, 3],    # Teilchen-Positionen
        "velocities": Tensor[N, 3],   # Geschwindigkeiten
        "masses": Tensor[N],          # Massen
        "potential": Tensor[N],       # Gravitationspotential
        "forces": Tensor[N, 3],       # Kräfte (optional)
        "meta": {
            "simulation_type": str,
            "time_step": float,
            "current_time": float,
            "units": Dict[str, str],
            "cosmology": Dict[str, float],
        }
    }
    """

    def __init__(self, positions: torch.Tensor, velocities: torch.Tensor,
                 masses: torch.Tensor, simulation_type: str = "nbody",
                 time_step: float = 0.01, current_time: float = 0.0,
                 cosmology: Optional[Dict[str, float]] = None, **kwargs):
        """
        Initialisiert SimulationTensorDict.

        Args:
            positions: [N, 3] Teilchen-Positionen
            velocities: [N, 3] Geschwindigkeiten
            masses: [N] Massen
            simulation_type: Art der Simulation
            time_step: Zeitschritt
            current_time: Aktuelle Zeit
            cosmology: Kosmologische Parameter
        """
        if positions.shape != velocities.shape:
            raise ValueError("Positions and velocities must have same shape")

        if positions.shape[0] != masses.shape[0]:
            raise ValueError("Number of positions must match number of masses")

        n_objects = positions.shape[0]

        # Default cosmology (Planck 2018)
        if cosmology is None:
            cosmology = {
                "H0": 67.4,      # Hubble constant
                "Omega_m": 0.315, # Matter density
                "Omega_L": 0.685, # Dark energy density
                "Omega_b": 0.049, # Baryon density
            }

        data = {
            "positions": positions,
            "velocities": velocities,
            "masses": masses,
            "potential": torch.zeros(n_objects),
            "forces": torch.zeros_like(positions),
            "meta": TensorDict({
                "simulation_type": simulation_type,
                "time_step": time_step,
                "current_time": current_time,
                "units": {
                    "length": "kpc",
                    "velocity": "km/s",
                    "mass": "solar_mass",
                    "time": "Myr"
                },
                "cosmology": cosmology,
            }, batch_size=(n_objects,))
        }

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    @property
    def positions(self) -> torch.Tensor:
        """Teilchen-Positionen."""
        return self["positions"]

    @property
    def velocities(self) -> torch.Tensor:
        """Teilchen-Geschwindigkeiten."""
        return self["velocities"]

    @property
    def masses(self) -> torch.Tensor:
        """Teilchen-Massen."""
        return self["masses"]

    def compute_gravitational_forces(self, softening: float = 0.1) -> torch.Tensor:
        """
        Berechnet Gravitationskräfte zwischen allen Teilchen.

        Args:
            softening: Softening-Parameter zur Vermeidung von Singularitäten

        Returns:
            [N, 3] Kraft-Tensor
        """
        positions = self.positions
        masses = self.masses
        n_particles = positions.shape[0]

        # Paarweise Distanzen
        r_ij = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 3]
        distances = torch.norm(r_ij, dim=-1)  # [N, N]

        # Softening anwenden
        distances = torch.sqrt(distances**2 + softening**2)

        # Gravitationskräfte: F = G * m1 * m2 / r^2 * r_hat
        G = 4.301e-6  # Gravitationskonstante in kpc * (km/s)^2 / solar_mass

        # Vermeidung von Selbst-Wechselwirkung
        mask = torch.eye(n_particles, dtype=torch.bool, device=positions.device)
        distances = distances.masked_fill(mask, float('inf'))

        # Kraft-Magnituden
        force_magnitudes = G * masses.unsqueeze(1) * masses.unsqueeze(0) / distances**2
        force_magnitudes = force_magnitudes.masked_fill(mask, 0)

        # Kraft-Richtungen
        force_directions = r_ij / distances.unsqueeze(-1)
        force_directions = force_directions.masked_fill(mask.unsqueeze(-1), 0)

        # Gesamtkräfte
        forces = -torch.sum(force_magnitudes.unsqueeze(-1) * force_directions, dim=1)

        self["forces"] = forces
        return forces

    def compute_potential_energy(self, softening: float = 0.1) -> torch.Tensor:
        """
        Berechnet Gravitationspotential für jedes Teilchen.

        Args:
            softening: Softening-Parameter

        Returns:
            [N] Potential-Tensor
        """
        positions = self.positions
        masses = self.masses
        n_particles = positions.shape[0]

        # Paarweise Distanzen
        r_ij = positions.unsqueeze(1) - positions.unsqueeze(0)
        distances = torch.norm(r_ij, dim=-1)

        # Softening anwenden
        distances = torch.sqrt(distances**2 + softening**2)

        # Selbst-Wechselwirkung vermeiden
        mask = torch.eye(n_particles, dtype=torch.bool, device=positions.device)
        distances = distances.masked_fill(mask, float('inf'))

        G = 4.301e-6  # Gravitationskonstante

        # Potential: Phi = -G * sum(m_j / r_ij)
        potential = -G * torch.sum(masses.unsqueeze(0) / distances, dim=1)

        self["potential"] = potential
        return potential

    def leapfrog_step(self) -> SimulationTensorDict:
        """
        Führt einen Leapfrog-Integrationsschritt durch.

        Returns:
            Neuer SimulationTensorDict mit aktualisiertem Zustand
        """
        dt = self["meta", "time_step"]

        # Berechne aktuelle Kräfte
        forces = self.compute_gravitational_forces()
        accelerations = forces / self.masses.unsqueeze(-1)

        # Leapfrog-Integration
        # v(t + dt/2) = v(t) + a(t) * dt/2
        half_step_velocities = self.velocities + accelerations * dt / 2

        # x(t + dt) = x(t) + v(t + dt/2) * dt
        new_positions = self.positions + half_step_velocities * dt

        # Erstelle neuen Zustand für Kraft-Berechnung
        temp_sim = SimulationTensorDict(
            positions=new_positions,
            velocities=half_step_velocities,
            masses=self.masses,
            simulation_type=self["meta", "simulation_type"],
            time_step=dt,
            current_time=self["meta", "current_time"],
            cosmology=self["meta", "cosmology"]
        )

        # Berechne neue Kräfte
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
            cosmology=self["meta", "cosmology"]
        )

    def compute_kinetic_energy(self) -> torch.Tensor:
        """Berechnet kinetische Energie."""
        velocities = self.velocities
        masses = self.masses

        # E_kin = 1/2 * m * v^2
        kinetic_energy = 0.5 * masses * torch.sum(velocities**2, dim=-1)
        return kinetic_energy

    def compute_total_energy(self) -> torch.Tensor:
        """Berechnet Gesamt-Energie des Systems."""
        kinetic = torch.sum(self.compute_kinetic_energy())
        potential = torch.sum(self.compute_potential_energy())
        return kinetic + potential

    def compute_center_of_mass(self) -> torch.Tensor:
        """Berechnet Schwerpunkt des Systems."""
        total_mass = torch.sum(self.masses)
        com = torch.sum(self.positions * self.masses.unsqueeze(-1), dim=0) / total_mass
        return com

    def compute_angular_momentum(self) -> torch.Tensor:
        """Berechnet Gesamt-Drehimpuls des Systems."""
        com = self.compute_center_of_mass()
        relative_positions = self.positions - com

        # L = sum(m_i * r_i x v_i)
        angular_momentum = torch.sum(
            self.masses.unsqueeze(-1) * torch.cross(relative_positions, self.velocities),
            dim=0
        )
        return angular_momentum

    def to_spatial_tensor(self) -> SpatialTensorDict:
        """Konvertiert zu SpatialTensorDict."""
        return SpatialTensorDict(
            coordinates=self.positions,
            coordinate_system="cartesian",
            unit=self["meta", "units", "length"]
        )

    def run_simulation(self, n_steps: int, save_interval: int = 1) -> List[SimulationTensorDict]:
        """
        Führt N-Body-Simulation über mehrere Zeitschritte aus.

        Args:
            n_steps: Anzahl der Zeitschritte
            save_interval: Speicher-Intervall

        Returns:
            Liste von SimulationTensorDict-Snapshots
        """
        snapshots = []
        current_sim = self

        for step in range(n_steps):
            if step % save_interval == 0:
                snapshots.append(current_sim)

            current_sim = current_sim.leapfrog_step()

            # Optional: Energie-Erhaltung prüfen
            if step % 100 == 0:
                energy = current_sim.compute_total_energy()
                print(f"Step {step}: Total Energy = {energy:.6f}")

        # Letzten Zustand hinzufügen
        snapshots.append(current_sim)

        return snapshots


class CosmologyTensorDict(AstroTensorDict):
    """
    TensorDict für kosmologische Berechnungen.

    Struktur:
    {
        "redshifts": Tensor[N],       # Rotverschiebungen
        "distances": TensorDict,      # Verschiedene Distanz-Maße
        "times": TensorDict,          # Zeit-Maße
        "meta": {
            "cosmology": Dict[str, float],
            "calculated_quantities": List[str],
        }
    }
    """

    def __init__(self, redshifts: torch.Tensor, 
                 cosmology: Optional[Dict[str, float]] = None, **kwargs):
        """
        Initialisiert CosmologyTensorDict.

        Args:
            redshifts: [N] Rotverschiebungen
            cosmology: Kosmologische Parameter
        """
        if cosmology is None:
            cosmology = {
                "H0": 67.4,      # km/s/Mpc
                "Omega_m": 0.315,
                "Omega_L": 0.685,
                "Omega_k": 0.0,
            }

        n_objects = redshifts.shape[0]

        data = {
            "redshifts": redshifts,
            "distances": TensorDict({}, batch_size=(n_objects,)),
            "times": TensorDict({}, batch_size=(n_objects,)),
            "meta": TensorDict({
                "cosmology": cosmology,
                "calculated_quantities": [],
            }, batch_size=(n_objects,))
        }

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    def compute_luminosity_distance(self) -> torch.Tensor:
        """Berechnet Leuchtkraft-Distanz."""
        z = self["redshifts"]
        cosmo = self["meta", "cosmology"]

        # Vereinfachte Berechnung für flaches Universum
        H0 = cosmo["H0"]
        Om = cosmo["Omega_m"]
        OL = cosmo["Omega_L"]

        # Komobile Distanz (vereinfacht)
        c = 299792.458  # km/s

        # Integral approximation
        a_values = torch.linspace(1/(1+z.max()), 1, 1000)
        integrand = 1 / torch.sqrt(Om * (1/a_values)**3 + OL)
        comoving_distance = c / H0 * torch.trapz(integrand, a_values)

        # Luminosity distance
        luminosity_distance = (1 + z) * comoving_distance

        self["distances", "luminosity"] = luminosity_distance
        self["meta", "calculated_quantities"].append("luminosity_distance")

        return luminosity_distance

    def compute_age_at_redshift(self) -> torch.Tensor:
        """Berechnet Alter des Universums bei gegebener Rotverschiebung."""
        z = self["redshifts"]
        cosmo = self["meta", "cosmology"]

        H0 = cosmo["H0"]
        Om = cosmo["Omega_m"]
        OL = cosmo["Omega_L"]

        # Hubble Zeit
        t_H = 9.78e9 / H0  # Gyr

        # Vereinfachte Alters-Berechnung
        age = t_H * torch.sqrt(OL / Om) * torch.log((1 + torch.sqrt(OL/Om)) / (torch.sqrt(OL/Om) + torch.sqrt(OL/Om + (1+z)**3)))

        self["times", "age"] = age
        self["meta", "calculated_quantities"].append("age_at_redshift")

        return age
