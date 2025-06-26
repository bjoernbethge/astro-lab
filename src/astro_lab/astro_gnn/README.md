# AstroGNN - Graph Neural Networks für Astronomische Punktwolken

Ein modulares Framework zur Analyse astronomischer Punktwolken mit PyTorch Geometric und TensorDict Integration.

## 🌟 Überblick

AstroGNN implementiert PointNet++ für die Klassifikation und Analyse astronomischer Punktwolken wie:
- Sternhaufen-Klassifikation
- Galaxienverteilungs-Analyse  
- Stellare Populations-Studien

## 📁 Projektstruktur

```
astro_gnn/
├── __init__.py          # Package Initialisierung
├── config.py            # Konfigurations-Management
├── model.py             # PointNet++ Modell
├── data.py              # Dataset und DataLoader
├── trainer.py           # PyTorch Lightning Trainer
├── cli.py               # Command Line Interface
├── configs/             # Beispiel-Konfigurationen
│   └── example_config.yaml
└── README.md           # Diese Datei
```

## 🚀 Schnellstart

### Installation

```bash
# Voraussetzungen
pip install torch torch-geometric pytorch-lightning tensordict
pip install rich click wandb h5py pandas
```

### Verwendung

1. **Erstelle eine Konfiguration:**
```bash
python -m astro_lab.astro_gnn create-config
```

2. **Trainiere das Modell:**
```bash
python -m astro_lab.astro_gnn train -c config.yaml
```

3. **Evaluiere:**
```bash
python -m astro_lab.astro_gnn evaluate -c config.yaml -ckpt checkpoints/best.ckpt
```

4. **Predictions:**
```bash
python -m astro_lab.astro_gnn predict -ckpt checkpoints/best.ckpt -i data/test -o predictions.csv
```

## 🏗️ Architektur

### Modell
- **PointNet++ Layers**: 3 hierarchische Feature-Extraktions-Layer
- **k-NN Graph**: Dynamische Graph-Konstruktion aus Punktwolken
- **Global Pooling**: Max-Pooling für Graph-Level Features
- **Klassifikations-Kopf**: MLP mit Dropout für Regularisierung

### Features
- **Input**: 3D-Koordinaten + optionale Features (Helligkeit, Farbe, etc.)
- **Output**: Klassifikation (z.B. Sterntypen O, B, A, F, G, K, M)

## 📊 Datenformat

Das Framework erwartet Daten im HDF5 oder CSV Format:

### HDF5 Struktur:
```
file.h5
├── cluster_0/
│   ├── positions [N, 3]  # x, y, z Koordinaten
│   ├── features [N, F]   # Zusätzliche Features
│   └── label            # Klassen-Label
├── cluster_1/
│   └── ...
```

### TensorDict Integration
```python
# Daten werden intern als TensorDict verarbeitet:
tensordict = TensorDict({
    "positions": torch.Tensor,  # [N, 3]
    "features": torch.Tensor,   # [N, F]
    "labels": torch.Tensor,     # [1]
    "batch": torch.Tensor       # Batch-Zuordnung
})
```

## 🔧 Konfiguration

Die YAML-Konfiguration kontrolliert alle Aspekte:

```yaml
model:
  input_features: 3
  hidden_dim: 128
  output_classes: 7
  k_neighbors: 16

training:
  epochs: 100
  learning_rate: 0.001
  scheduler: "cosine"
```

## 📈 Metriken

- **Accuracy**: Gesamt-Genauigkeit
- **F1-Score**: Macro-averaged F1
- **Confusion Matrix**: Automatisch gespeichert
- **Learning Curves**: Via TensorBoard

## 🎓 Für die Abschlussarbeit

### Kernkonzepte:
1. **Graph-Konstruktion**: k-NN aus Punktwolken
2. **Message Passing**: PointNet++ Aggregation
3. **Hierarchische Features**: Multi-Scale Analyse
4. **TensorDict**: Flexible Daten-Container

### Erweiterungsmöglichkeiten:
- Temporal Features für Zeitreihen
- Multi-Modal Fusion (Spektren + Positionen)
- Domain Adaptation zwischen Surveys
- Unsupervised Pre-Training

## 📝 Zitierung

Wenn Sie dieses Framework verwenden:
```bibtex
@software{astrognn2024,
  title={AstroGNN: Graph Neural Networks for Astronomical Point Clouds},
  author={[Ihr Name]},
  year={2024},
  url={https://github.com/yourusername/astro-lab}
}
```
