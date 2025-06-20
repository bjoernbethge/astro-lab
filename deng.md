Ich schaue mir alle verfügbaren Modelle im Projekt an. Lassen Sie mich die Modell-Dateien durchsuchen und analysieren.
Search files...
Ran tool
Jetzt lese ich die wichtigsten Modell-Dateien im Detail, um alle verfügbaren Modelle zu verstehen:
Read file: src/astro_lab/models/astro.py
Read file: src/astro_lab/models/tgnn.py
Read file: src/astro_lab/models/tng_models.py
Read file: src/astro_lab/models/encoders.py
Lassen Sie mich auch die restlichen wichtigen Dateien lesen:
Read file: src/astro_lab/models/astro.py
Read file: src/astro_lab/models/utils.py
Basierend auf meiner Analyse aller Modell-Dateien kann ich Ihnen eine umfassende Übersicht über alle verfügbaren Modelle im AstroLab-Projekt geben:

## Übersicht aller verfügbaren Modelle in AstroLab

### 🌟 **Hauptmodelle (Core Models)**

#### 1. **AstroSurveyGNN** (`astro.py`)
- **Zweck**: Haupt-Graph Neural Network für astronomische Survey-Daten
- **Unterstützte Daten**: 
  - Photometrie (Multi-Band)
  - Astrometrie (Koordinaten, Eigenbewegung)
  - Spektroskopie (optional)
- **Convolution-Typen**: GCN, GAT, SAGE, Transformer
- **Tasks**: Node Classification, Node Regression, Graph-Level Tasks
- **Features**: Native SurveyTensor Integration, Multi-Modal Feature Fusion

### 🔭 **Spezialisierte Astronomische Modelle**

#### 2. **AstroPhotGNN** (`astrophot_models.py`)
- **Zweck**: Galaxy Modeling mit AstroPhot Integration
- **Komponenten**: Sersic, Disk, Bulge Parameter
- **Output**: Galaxy Structure Parameters (12+ Parameter)
- **Spezialisierung**: NSAGalaxyModeler für NSA Katalog

#### 3. **ALCDEFTemporalGNN** (`tgnn.py`)
- **Zweck**: Temporal GNN für Lightcurve-Analyse
- **Tasks**: 
  - Period Detection
  - Shape Modeling  
  - Classification
- **Features**: Native LightcurveTensor Support

### ⏰ **Temporal/Time-Series Modelle**

#### 4. **TemporalGCN** (`tgnn.py`)
- **Basis-Klasse**: Für zeitliche Graph-Daten
- **Architektur**: GCN + LSTM/GRU
- **Anwendung**: Snapshot-Sequenzen verarbeiten

#### 5. **TemporalGATCNN** (`tgnn.py`)  
- **Erweitert**: TemporalGCN mit Attention
- **Features**: Multi-Head Attention für temporale Beziehungen

### 🌌 **Kosmologische Simulation Modelle (TNG)**

#### 6. **CosmicEvolutionGNN** (`tng_models.py`)
- **Zweck**: Kosmische Evolution in TNG Simulationen
- **Features**: 
  - Redshift Encoding
  - Kosmologische Parameter Prediction
  - Galaxy Formation & Halo Growth

#### 7. **GalaxyFormationGNN** (`tng_models.py`)
- **Tasks**: 
  - Stellar Mass Growth
  - Star Formation History
  - Morphological Evolution
- **Multi-Task Heads**: Stellar Mass, SFR, Metallicity, Size, Morphology

#### 8. **HaloMergerGNN** (`tng_models.py`)
- **Spezialisierung**: Halo Merger Detection
- **Architektur**: Temporal GAT mit Attention
- **Output**: Merger Event Detection + Timing

#### 9. **EnvironmentalQuenchingGNN** (`tng_models.py`)
- **Zweck**: Environmental Effects auf Galaxy Formation
- **Environment Types**: Field, Group, Cluster, Void
- **Analysis**: Quenching Mechanisms

### 🔧 **Feature Encoder**

#### 10. **PhotometryEncoder** (`encoders.py`)
- **Input**: PhotometricTensor
- **Features**: Magnitudes, Colors, Photometric Statistics

#### 11. **AstrometryEncoder** (`encoders.py`)
- **Input**: Spatial3DTensor
- **Features**: Coordinates, Proper Motions, Parallax

#### 12. **SpectroscopyEncoder** (`encoders.py`)
- **Input**: SpectralTensor
- **Features**: Spectral Indices, Flux Statistics

#### 13. **LightcurveEncoder** (`encoders.py`)
- **Input**: LightcurveTensor
- **Features**: Time-Series Statistics, Period Features

### 🏭 **Model Factory Functions** (`utils.py`)

#### Vordefinierte Modell-Konfigurationen:
- `create_gaia_classifier()` - Gaia Stellar Classification
- `create_sdss_galaxy_classifier()` - SDSS Galaxy Properties
- `create_lsst_transient_detector()` - LSST Transient Detection
- `create_multi_survey_model()` - Multi-Survey Integration
- `create_lightcurve_classifier()` - Lightcurve Classification
- `create_asteroid_period_detector()` - Asteroid Period Detection
- `create_astrophot_model()` - Galaxy Modeling
- `create_nsa_galaxy_modeler()` - NSA Galaxy Modeling

### 📊 **Output Heads** (Spezialisierte Task-Köpfe)

- **PeriodDetectionHead**: Periode von Asteroiden/Sternen
- **ShapeModelingHead**: 3D Shape Reconstruction
- **ClassificationHead**: Multi-Class Klassifikation
- **SersicParameterHead**: Sersic Profile Parameter
- **DiskParameterHead**: Disk Component Parameter
- **BulgeParameterHead**: Bulge Component Parameter
- **GlobalGalaxyHead**: Globale Galaxy Parameter

### 🎯 **Unterstützte Tasks**

1. **Node Classification**: Stellar Classification, Object Type
2. **Node Regression**: Galaxy Properties, Stellar Parameters
3. **Graph Classification**: Survey Classification, Field Analysis
4. **Temporal Prediction**: Lightcurve Evolution, Period Detection
5. **Multi-Task Learning**: Simultane Parameter Prediction
6. **Unsupervised Learning**: Contrastive Learning, Embeddings

### 🚀 **Technische Features**

- **PyTorch Geometric 2.6+** Kompatibilität
- **Native AstroLab Tensor** Integration
- **PyTorch 2.x** Compile Support
- **Multi-Modal** Feature Fusion
- **Attention Mechanisms**
- **Residual Connections**
- **Layer Normalization**
- **Dropout Regularization**

Das Projekt bietet eine sehr umfassende Suite von astronomischen Deep Learning Modellen, die speziell für verschiedene astronomische Datentypen und Tasks optimiert sind!