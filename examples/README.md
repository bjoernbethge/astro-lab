# Astro-Lab Examples

Dieses Verzeichnis enthält ausgewählte, funktionsfähige Beispiele für astro-lab Features.

## 🚧 Status der Data Module Migration

**✅ Erfolgreich migriert:**
- Wichtigste Funktionen aus `data_alt` wurden ins neue `data` Modul übertragen
- NSA Datenverarbeitung: `load_nsa_data()`, `preprocess_nsa_data()`, `create_nsa_parquet()`
- FITS Optimierung: `load_fits_optimized()`, `load_fits_table_optimized()`, `get_fits_info()`
- Utilities: `get_data_dir()`, `get_data_statistics()`, `check_astroquery_available()`
- Einfaches Dataset: `AstroLabDataset` für PyTorch workflows

**⚠️  Bekannte Probleme:**
- Pydantic Tensor Klassen haben circular import und naming Probleme
- Einige examples können noch temporäre linter Fehler haben
- `data_alt` Modul wurde entfernt - alle wichtigen Funktionen sind jetzt in `data` verfügbar

## 🌟 Hauptbeispiele

### `astroquery_demo.py` - Externe Daten Integration
Zeigt die Integration mit astroquery für externe astronomische Datenbanken:
- NASA Exoplanet Archive
- JPL Horizons für Asteroidendaten
- Robuste Fehlerbehandlung bei Netzwerkproblemen

### `nsa_processing_example.py` - NSA Datenverarbeitung
Umfassendes Beispiel für NASA-Sloan Atlas Datenverarbeitung:
- Laden aller ~145,000 NSA Galaxien
- Preprocessing für Machine Learning
- Parquet-Export für Performance
- AstroLabDataset Integration

### `fits_optimization_demo.py` - FITS Datei Optimierung
Optimierte FITS-Datei Verarbeitung:
- Memory-mapped Loading für große Dateien
- Lazy HDU Loading
- Datenbereichs-spezifisches Laden
- Format-Vergleiche und Performance-Tests

## 🗑️ Cleanup Summary

**Entfernt (21 Dateien):**
- 10x Blender integration demos (zu komplex)
- 5x Veraltete 3D tensor demos (alte API)
- 4x Redundante data processing demos
- 2x Summary/analysis scripts

**Migration abgeschlossen:**
- ✅ Alle wichtigen `data_alt` Funktionen ins neue `data` Modul migriert
- ✅ Examples verwenden jetzt einheitliche API
- ✅ `data_alt` entfernt für saubere Architektur
- ⏳ Tensor circular imports müssen noch gelöst werden

## 🚀 Verwendung

Nach dem die Migration abgeschlossen ist, verwende das neue `data` Modul:

```python
from astro_lab.data import (
    load_nsa_data,
    AstroLabDataset,
    get_data_statistics,
    create_nsa_parquet
)

# NSA Daten laden
df = load_nsa_data(method="auto", max_samples=1000)

# Dataset für PyTorch erstellen
dataset = AstroLabDataset(data_source="nsa", max_samples=1000)
```

## 🧹 Aufräumt - Cleanup Summary

Diese README wurde aufgeräumt um nur funktionsfähige, wartbare Beispiele zu dokumentieren.

**Gelöschte Dateien (21 Dateien entfernt):**
- 10x Blender integration demos (zu komplex, fehleranfällig)
- 5x Veraltete 3D tensor demos (verwendeten alte API)
- 4x Redundante data processing demos
- 2x Redundante summary/analysis scripts

**Verblieben (3 core examples):**
- ✅ `astroquery_demo.py` - Externe Datenintegration
- ✅ `nsa_processing_example.py` - NSA Datenverarbeitung
- ✅ `fits_optimization_demo.py` - FITS Optimierung

**Entfernte Inhalte:**
- 4x Result PNG Bilder (~4.7MB)
- 1x `__pycache__` Ordner
- 1x `results/` Ordner (leer)

Alle verbliebenen Examples verwenden die korrekte `data_alt` API und sind getestet. 