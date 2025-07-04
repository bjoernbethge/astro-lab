# Survey Preprocessing Overview

## Einheitlicher Preprocessing-Prozess

Alle Surveys in astro-lab folgen einem einheitlichen Preprocessing-Prozess:

### 1. **Datenpipeline**
```
Raw Data → Collector → Preprocessor → TensorDict → Graph → Dataset
```

### 2. **Verfügbare Surveys**

| Survey | Status | Collector | Preprocessor | Image Support |
|--------|--------|-----------|--------------|---------------|
| **Gaia** | ✅ Complete | ✅ | ✅ | ❌ |
| **SDSS** | ✅ Complete | ✅ | ✅ | ❌ |
| **NSA** | ✅ Complete | ✅ | ✅ | ✅ (FITS) |
| **Exoplanet** | ✅ Complete | ✅ | ✅ | ❌ |
| **TNG50** | ✅ Complete | ✅ | ✅ | ❌ |
| **2MASS** | ✅ Complete | ✅ | ✅ | ❌ |
| **WISE** | ✅ Complete | ✅ | ✅ | ❌ |
| **Pan-STARRS** | ✅ Complete | ✅ | ✅ | ❌ |
| **DES** | ✅ Complete | ✅ | ✅ | ❌ |
| **Euclid** | ✅ Complete | ✅ | ✅ | ❌ |

### 3. **Einheitliche Preprocessor-Struktur**

Alle Preprocessors erben von `BaseSurveyProcessor` und implementieren:

```python
class SurveyPreprocessor(BaseSurveyProcessor):
    def __init__(self):
        super().__init__("survey_name")
    
    def extract_coordinates(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract spatial coordinates"""
        
    def extract_features(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract feature vector"""
        
    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Survey-specific preprocessing"""
```

### 4. **Standardisierte Features**

**Alle Surveys extrahieren:**
- **Koordinaten**: RA/Dec/Distance → 3D Cartesian (x, y, z) in Parsec
- **Photometrie**: Magnituden in verschiedenen Bändern
- **Farben**: Automatische Berechnung von Farbindizes
- **Qualitätsflags**: Filterung nach Datenqualität
- **Fehler**: Magnitudenfehler und Signal-to-Noise Ratios

**3D-Koordinaten-Strategien:**
- **Gaia**: Parallax → Distance → 3D Cartesian
- **SDSS**: Redshift → Comoving Distance → 3D Cartesian
- **NSA**: Redshift → Comoving Distance → 3D Cartesian
- **2MASS**: Magnitude → Estimated Distance → 3D Cartesian
- **WISE**: Magnitude → Estimated Distance → 3D Cartesian
- **Pan-STARRS**: Magnitude → Estimated Distance → 3D Cartesian
- **DES**: Magnitude → Estimated Distance → 3D Cartesian
- **Euclid**: Magnitude → Estimated Distance → 3D Cartesian
- **TNG50**: Bereits 3D Cartesian (Simulation)
- **Exoplanet**: Host Star Distance → 3D Cartesian

**Survey-spezifische Features:**
- **Gaia**: Parallax, Eigenbewegung, BP-RP Farbe
- **SDSS**: u-g-r-i-z Bänder, Spektroskopie
- **NSA**: Galaxy Properties, Concentration Index, Log Mass
- **2MASS**: J-H-K Bänder, Near-IR Photometrie
- **WISE**: W1-W2-W3-W4 Bänder, Mid-IR
- **Pan-STARRS**: g-r-i-z-y Bänder, Deep Imaging
- **DES**: g-r-i-z-Y Bänder, Weak Lensing
- **Euclid**: VIS-Y-J-H Bänder, Shear Measurements

### 5. **TensorDict Integration**

Alle Preprocessors erstellen einheitliche `SurveyTensorDict`:

```python
tensor_dict = SurveyTensorDict(
    spatial=spatial_tensor,      # Koordinaten
    photometric=photometric_tensor,  # Magnituden
    features=features_tensor,    # Zusätzliche Features
    images=image_tensor,         # Optional (nur NSA)
    survey_name=survey_name,
    data_release=data_release
)
```

### 6. **Graph Building**

**Einheitlicher Graph-Building-Prozess:**
- **k-NN Graph**: Basierend auf räumlichen Koordinaten
- **Feature Integration**: Alle TensorDict-Komponenten werden zu Node-Features
- **Caching**: Graphen werden als `.pt` Dateien gespeichert
- **Automatisch**: Während des Trainings oder Dataset-Erstellung

### 7. **Qualitätskontrolle**

**Standardisierte Qualitätsfilter:**
- **Magnituden**: Filterung von extremen Werten (< -999)
- **Qualitätsflags**: Survey-spezifische Qualitätskriterien
- **Signal-to-Noise**: Mindest-SNR für zuverlässige Detektionen
- **Koordinaten**: Validierung der RA/Dec Werte

### 8. **Verwendung**

**Einheitliche API für alle Surveys:**

```python
from astro_lab.config import get_preprocessor

# Preprocessor erstellen
preprocessor = get_preprocessor("gaia")

# Daten laden und verarbeiten
df = preprocessor.load_data(max_samples=1000)
tensor_dict = preprocessor.to_tensordict(df)

# Graph erstellen
from astro_lab.data.graphs import create_knn_graph
graph = create_knn_graph(tensor_dict, k_neighbors=8)
```

### 9. **Erweiterbarkeit**

**Neue Surveys hinzufügen:**
1. Collector in `collectors/` erstellen
2. Preprocessor in `preprocessors/` erstellen
3. Registry in `__init__.py` erweitern
4. Konfiguration in `config/surveys.py` hinzufügen

### 10. **Image Support**

**Aktuell nur NSA:**
- **FITS Images**: Automatisches Laden und Normalisieren
- **ImageTensorDict**: Neue TensorDict-Klasse für Bilddaten
- **Integration**: Images werden in SurveyTensorDict eingebaut
- **Erweiterbar**: Andere Surveys können Image-Support hinzufügen

### 11. **Performance**

**Optimierungen:**
- **Lazy Loading**: Daten werden nur bei Bedarf geladen
- **Caching**: Verarbeitete Daten und Graphen werden gespeichert
- **Memory Management**: Automatische Bereinigung nach Verarbeitung
- **Parallel Processing**: Unterstützung für Multi-Processing

### 12. **Monitoring**

**Logging und Metriken:**
- **Progress Tracking**: Fortschritt bei Datenverarbeitung
- **Quality Metrics**: Datenqualitätsstatistiken
- **Memory Usage**: Speicherverbrauch während Verarbeitung
- **Error Handling**: Robuste Fehlerbehandlung

---

**Fazit:** Das System ist vollständig einheitlich und erweiterbar. Alle 10 Surveys folgen dem gleichen Prozess und können nahtlos in die ML-Pipeline integriert werden. 