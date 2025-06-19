# Astronomische Daten laden & verarbeiten

## 🎯 Was macht dieses Modul?

Das `astro_lab.data` Modul lädt astronomische Kataloge (Sterne, Galaxien) und macht sie bereit für Machine Learning. **Einfach, schnell, ohne Komplexität.**

## 📋 Kompletter Workflow

### 1️⃣ Daten laden (30 Sekunden)

```python
from astro_lab.data import load_gaia_data, load_sdss_data

# Gaia Sterne laden
sterne = load_gaia_data(max_samples=5000)
print(f"✅ {sterne.shape[0]} Sterne geladen")

# SDSS Galaxien laden  
galaxien = load_sdss_data(max_samples=2000)
print(f"✅ {galaxien.shape[0]} Galaxien geladen")
```

### 2️⃣ Daten anschauen

```python
# Was ist drin?
print(f"Gaia Features: {sterne.photometric_bands}")  # ['G', 'BP', 'RP']
print(f"SDSS Features: {galaxien.photometric_bands}")  # ['u', 'g', 'r', 'i', 'z']

# Erste Zeilen
print(sterne.data[:3])  # Erste 3 Sterne
print(galaxien.data[:3])  # Erste 3 Galaxien
```

### 3️⃣ Weiterverarbeitung für ML

```python
# Für PyTorch Training
import torch
from torch.utils.data import DataLoader

# Als normaler PyTorch Tensor
X = sterne.data  # [5000, 8] - 5000 Sterne, 8 Features
y = torch.randint(0, 3, (5000,))  # Dummy Labels

# DataLoader für Training
loader = DataLoader(
    list(zip(X, y)), 
    batch_size=32, 
    shuffle=True
)

# Training Loop
for batch_x, batch_y in loader:
    # Hier kommt dein ML Model
    pass
```

### 4️⃣ Für Graph Neural Networks

```python
from astro_lab.data import AstroDataset

# Graph Dataset erstellen
dataset = AstroDataset(
    survey="gaia",
    max_samples=1000,
    k_neighbors=8  # 8 nächste Nachbarn
)

# Graph anschauen
graph = dataset[0]
print(f"Graph: {graph.num_nodes} Knoten, {graph.num_edges} Kanten")

# Für PyTorch Geometric
from torch_geometric.loader import DataLoader
graph_loader = DataLoader([graph], batch_size=1)
```

## 🗂️ Verfügbare Daten

| Survey | Was | Anzahl | Features | Verwendung |
|--------|-----|--------|----------|------------|
| **Gaia** | Sterne | 5k-50k | Position, Helligkeit, Bewegung | Stellar Classification |
| **SDSS** | Galaxien | 1k-10k | Farben, Redshift, Morphologie | Galaxy Classification |
| **NSA** | Galaxien | 1k-5k | Sérsic Profile, Masse | Galaxy Evolution |
| **LINEAR** | Asteroiden | 500-2k | Lichtkurven, Perioden | Variable Stars |

## 🚀 Schnellstart-Rezepte

### Rezept 1: Stellar Classification

```python
# 1. Daten laden
from astro_lab.data import load_gaia_data
sterne = load_gaia_data(max_samples=10000)

# 2. Features extrahieren (G, BP, RP Magnitudes)
magnitudes = sterne.data[:, 5:8]  # Spalten 5,6,7 sind G,BP,RP
farben = magnitudes[:, 1] - magnitudes[:, 2]  # BP-RP Farbe

# 3. Einfache Klassifikation
import numpy as np
# Rote Riesen (BP-RP > 1.0), Hauptreihe (0.5-1.0), Blaue Sterne (<0.5)
labels = np.where(farben > 1.0, 2, np.where(farben > 0.5, 1, 0))

# 4. Training/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    magnitudes, labels, test_size=0.2
)

print(f"Training: {len(X_train)}, Test: {len(X_test)}")
```

### Rezept 2: Galaxy Morphology

```python
# 1. Galaxien laden
from astro_lab.data import load_sdss_data
galaxien = load_sdss_data(max_samples=5000)

# 2. Farben berechnen (g-r, r-i)
mags = galaxien.data[:, 4:7]  # g,r,i Magnitudes
g_r = mags[:, 0] - mags[:, 1]  # g-r
r_i = mags[:, 1] - mags[:, 2]  # r-i

# 3. Morphologie-Features
features = np.column_stack([g_r, r_i, galaxien.data[:, 2]])  # + Redshift

# 4. Elliptical (rot), Spiral (blau) Trennung
# Ellipticals: g-r > 0.7, Spirals: g-r < 0.7
morphology = (g_r > 0.7).astype(int)

print(f"Ellipticals: {np.sum(morphology)}, Spirals: {len(morphology) - np.sum(morphology)}")
```

### Rezept 3: Lightning Training

```python
# 1. DataModule erstellen
from astro_lab.data import AstroDataModule
import lightning as L

datamodule = AstroDataModule(
    survey="gaia",
    batch_size=64,
    max_samples=20000,
    train_ratio=0.7,
    val_ratio=0.15
)

# 2. Einfaches Model
class StellarClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 64),  # 8 Gaia Features
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)   # 3 Stellar Classes
        )
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # Hier deine Loss Function
        pass

# 3. Training
model = StellarClassifier()
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, datamodule)
```

## 🔧 Häufige Probleme & Lösungen

### Problem: "Keine echten Daten gefunden"
```python
# Lösung: Demo-Daten werden automatisch generiert
dataset = load_gaia_data(max_samples=1000)
# ✅ Funktioniert immer, auch ohne Internet
```

### Problem: "Zu langsam bei großen Datasets"
```python
# Lösung: Weniger Samples verwenden
quick_data = load_gaia_data(max_samples=1000)  # Statt 50000
```

### Problem: "Falsche Feature-Dimensionen"
```python
# Lösung: Shape checken
data = load_gaia_data(max_samples=100)
print(f"Shape: {data.shape}")  # [100, 8]
print(f"Features: {data.column_mapping}")  # Welche Spalte ist was
```

## 📊 Was passiert intern?

### Demo-Daten Generation
Wenn keine echten Kataloge da sind:

```python
# Gaia Demo-Daten (realistisch)
ra = np.random.uniform(0, 360, n_stars)      # Himmelskoordinaten
dec = np.random.uniform(-90, 90, n_stars)
g_mag = np.random.normal(12, 2, n_stars)     # G Magnitude ~12±2
bp_rp = np.random.normal(0.8, 0.3, n_stars) # BP-RP Farbe
parallax = np.random.exponential(2, n_stars) # Parallaxe
```

### Graph Construction
Für GNN Training:

```python
# k-NN Graph aus Himmelskoordinaten
from sklearn.neighbors import NearestNeighbors
coords = np.column_stack([ra, dec])
nbrs = NearestNeighbors(n_neighbors=8)
distances, indices = nbrs.fit(coords).kneighbors(coords)
# → Edge List für PyTorch Geometric
```

## 🎓 Nächste Schritte

### Nach dem Laden → ML Pipeline

```python
# 1. Daten geladen ✅
data = load_gaia_data(max_samples=5000)

# 2. Feature Engineering
def create_features(tensor_data):
    # Farben berechnen
    g = tensor_data[:, 5]
    bp = tensor_data[:, 6] 
    rp = tensor_data[:, 7]
    
    features = {
        'colors': bp - rp,
        'absolute_mag': g - 5 * np.log10(tensor_data[:, 2]) + 5,  # Mit Parallaxe
        'proper_motion': np.sqrt(tensor_data[:, 3]**2 + tensor_data[:, 4]**2)
    }
    return np.column_stack(list(features.values()))

features = create_features(data.data)

# 3. Model Training
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
# model.fit(features, labels)  # Deine Labels
```

### Integration mit anderen Modulen

```python
# Mit astro_lab.models
from astro_lab.models import ModelFactory

data = load_gaia_data(max_samples=5000)
model = ModelFactory.create_model("stellar_classification", survey="gaia")
# → Automatisch passendes Model für Gaia Daten

# Mit astro_lab.training  
from astro_lab.training import LightningModule
trainer = LightningModule(model, data)
trainer.fit()
```

## 📝 Development Commands

```bash
# Schnelltest
uv run python -c "from astro_lab.data import load_gaia_data; print('✅ Works')"

# Alle Surveys testen
uv run python -c "
from astro_lab.data import SURVEY_CONFIGS
for survey in SURVEY_CONFIGS.keys():
    print(f'✅ {survey} available')
"

# Performance Test
uv run python -c "
import time
from astro_lab.data import load_gaia_data
start = time.time()
data = load_gaia_data(max_samples=1000)
print(f'⚡ Loaded in {time.time()-start:.2f}s')
"
```

---

**TL;DR**: `load_gaia_data()` → Features extrahieren → ML Model trainieren → Fertig! 🚀 