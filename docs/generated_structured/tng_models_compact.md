# Tng_Models Module

Auto-generated documentation for `astro_lab.models.tng_models`

## Classes

### CosmicEvolutionGNN

Temporal GNN for cosmic evolution in TNG simulations.
Analyzes galaxy formation, halo growth, and large-scale structure.

#### Methods

**`encode_snapshot_with_redshift(self, x: torch.Tensor, edge_index: torch.Tensor, redshift: float, batch: Optional[torch.Tensor] = None) -> torch.Tensor`**

Encode snapshot with redshift information.

**`forward(self, snapshot_sequence: List[Dict[str, torch.Tensor]], redshifts: Optional[List[float]] = None) -> Dict[str, torch.Tensor]`**

Process temporal sequence with cosmological time encoding.

### EnvironmentalQuenchingGNN

Temporal GNN for environmental quenching analysis.
Studies how large-scale environment affects star formation.

#### Methods

**`encode_environmental_effects(self, embeddings: torch.Tensor) -> torch.Tensor`**

Encode environmental effects using attention.

**`forward(self, snapshot_sequence: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]`**

Forward pass for environmental quenching analysis.

### GalaxyFormationGNN

Temporal GNN for galaxy formation and evolution.
Predicts stellar mass growth, star formation history, and morphological evolution.

#### Methods

**`predict_galaxy_properties(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]`**

Predict galaxy properties from embeddings.

**`forward(self, snapshot_sequence: List[Dict[str, torch.Tensor]], predict_properties: bool = True) -> Dict[str, torch.Tensor]`**

Forward pass for galaxy formation analysis.

### HaloMergerGNN

Temporal GNN for halo merger analysis using attention mechanisms.

#### Methods

**`detect_merger_events(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`**

Detect merger events from temporal embeddings.

**`forward(self, snapshot_sequence: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]`**

Forward pass with merger detection.
