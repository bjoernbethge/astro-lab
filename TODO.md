# TODO - Vollständige Problemanalyse AstroLab

## 🔥 KRITISCHE PROBLEME (SOFORT BEHEBEN)

### 1. CLI komplett kaputt
- **Problem**: `src/astro_lab/cli/__init__.py` importiert nicht-existierende Funktionen
- **Details**: 
  - Zeile 25-36: `load_gaia_data`, `load_nsa_data`, `load_sdss_data`, `load_tng50_data` existieren nicht mehr
  - Zeile 432: `astro_processing_context` ist nicht definiert
  - Entry Point in `pyproject.toml` zeigt auf kaputte Datei
- **Fix**: 
  - [ ] Nicht-existierende Imports entfernen
  - [ ] `astro_processing_context` definieren oder entfernen
  - [ ] Entry Point korrigieren

### 2. Environment Installation (FAST REPARIERT)
- **Problem**: Virtual Environment war korrupt
- **Status**: 🟡 Fast repariert - nur noch kleine Probleme
- **Details**:
  - ~~`kiwisolver` Paket hat keine `__version__` Attribute~~ ✅ Behoben
  - ~~Dependency-Konflikte durch meine Änderungen~~ ✅ Behoben
  - ~~Installation schlägt fehl~~ ✅ Behoben
- **Verbleibende Aufgaben**:
  - [ ] Letzte kleinere venv-Probleme beheben

### 3. DataModule Auto-Setup Rekursion
- **Problem**: Unnötige `self.setup()` Aufrufe die ich hinzugefügt habe
- **Details**:
  - `src/astro_lab/data/datamodule.py` Zeile 95, 113, 131
  - Lightning ruft `setup()` automatisch auf
  - Kann zu Rekursion führen
- **Fix**:
  - [ ] Auto-setup Logik entfernen
  - [ ] Zurück zu Lightning Standard
  - [ ] Nur ValueError wenn Dataset leer

## 🧮 MATHEMATISCHE UND ASTRONOMISCHE FEHLER

### 4. Koordinaten-Transformationen fehlerhaft
- **Problem**: `src/astro_lab/tensors/spatial_3d.py` hat astronomische Ungenauigkeiten
- **Details**:
  - Zeile 249-251: `torch.rad2deg(ra) % 360.0` - RA sollte 0-360° sein, aber % kann negative Werte erzeugen
  - Zeile 252: `torch.rad2deg(dec)` - Dec sollte auf ±90° begrenzt werden
  - Zeile 245: `torch.asin(torch.clamp(z / torch.clamp(distance, min=1e-8), -1.0, 1.0))` - Numerisch instabil bei kleinen Distanzen
  - Zeile 414: `coords_rad = np.radians(coords)` - Preprocessing nutzt Haversine aber ohne Validierung der Eingabe-Einheiten
- **Fix**:
  - [ ] RA-Normalisierung korrigieren: `(torch.rad2deg(ra) + 360.0) % 360.0`
  - [ ] Dec-Clipping hinzufügen: `torch.clamp(torch.rad2deg(dec), -90.0, 90.0)`
  - [ ] Numerische Stabilität für kleine Distanzen verbessern
  - [ ] Einheiten-Validierung in Preprocessing hinzufügen

### 5. NaN-Handling inkonsistent und gefährlich
- **Problem**: NaN-Werte werden überall unterschiedlich behandelt
- **Details**:
  - `src/astro_lab/data/preprocessing.py` Zeile 1228: `np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)` - Ersetzt NaN mit 0, aber das verfälscht astronomische Daten
  - Zeile 456, 507, 548, 589, 659: `torch.nan_to_num(features, nan=0.0)` - Inkonsistent, manchmal ohne posinf/neginf
  - `src/astro_lab/tensors/base.py` Zeile 81: `torch.isfinite(data).all()` - Wirft Fehler aber behandelt nicht, was dann passiert
- **Fix**:
  - [ ] Einheitliche NaN-Behandlung definieren
  - [ ] NaN mit astronomisch sinnvollen Werten ersetzen (z.B. Median, nicht 0)
  - [ ] Warnung ausgeben wenn NaN-Werte gefunden werden
  - [ ] Dokumentieren welche Spalten NaN haben dürfen

### 6. Magnitude-Berechnungen mathematisch falsch
- **Problem**: Farb-Indizes und Magnitudes werden falsch berechnet
- **Details**:
  - `src/astro_lab/data/preprocessing.py` Zeile 461: `bp_rp = np.nan_to_num(bp_rp, nan=0.0)` - Farb-Index 0.0 ist astronomisch unsinnig
  - Zeile 462-468: Stellar classification basiert auf willkürlichen Bins ohne astronomische Grundlage
  - `src/astro_lab/tensors/statistics.py` Zeile 157: `valid_mask = torch.isfinite(magnitudes)` - Aber keine Behandlung ungültiger Magnitudes
- **Fix**:
  - [ ] Farb-Indizes mit Median der Sterne gleichen Typs ersetzen
  - [ ] Stellar classification auf echte astronomische Kategorien basieren
  - [ ] Magnitude-Limits definieren (z.B. -10 bis +30)

### 7. Distanz-Berechnungen fehlerhaft
- **Problem**: Verschiedene Distanz-Metriken werden inkonsistent verwendet
- **Details**:
  - `src/astro_lab/data/preprocessing.py` Zeile 825-839: Haversine-Formel für Himmelkoordinaten korrekt, aber...
  - Zeile 414: `coords_rad = np.radians(coords)` - Annahme dass Input in Grad ist, aber nicht validiert
  - `src/astro_lab/tensors/spatial_3d.py` Zeile 614-618: Unit-Konvertierung hardcoded (kpc->pc, Mpc->pc) ohne Validierung
- **Fix**:
  - [ ] Einheiten-Validierung vor allen Distanz-Berechnungen
  - [ ] Konsistente Distanz-Metrik definieren (Haversine für Himmel, Euklidisch für 3D)
  - [ ] Unit-Konvertierung in separate, getestete Funktionen

## 🏗️ ARCHITEKTUR-PROBLEME

### 8. Preprocessing Doppelungen und Chaos
- **Problem**: Redundante Funktionen die ich hinzugefügt habe
- **Details**:
  - `src/astro_lab/data/preprocessing.py` ist doppelt so lang geworden (1377 Zeilen)
  - `create_standardized_files()` und `process_survey()` überlappen mit existierenden
  - Verwirrende Redundanz zwischen alter und neuer API
- **Fix**:
  - [ ] Neue redundante Funktionen entfernen
  - [ ] Datei auf ursprüngliche Länge reduzieren
  - [ ] Nur eine saubere API behalten

### 9. Dataset Single-Graph Handling fehlerhaft
- **Problem**: Linter-Warnungen und Type-Probleme durch meine Änderungen
- **Details**:
  - `src/astro_lab/datasets/astro_dataset.py` Zeile 131-155
  - Funktionalität ist OK, aber Typen-Probleme
  - Linter beschwert sich über Dataset-Attribute die nicht existieren
- **Fix**:
  - [ ] Type Hints korrigieren
  - [ ] Linter-Warnungen beheben ohne Funktionalität zu brechen

### 10. NotImplementedError überall
- **Problem**: Viele Funktionen sind nicht implementiert
- **Details**:
  - `src/astro_lab/data/core.py` Zeile 853: `process()` nicht implementiert
  - `src/astro_lab/tensors/orbital.py` Zeile 354: Propagator nicht implementiert
  - `src/astro_lab/tensors/earth_satellite.py` Zeile 637: Satellite tracking nicht implementiert
- **Fix**:
  - [ ] Alle NotImplementedError finden und bewerten
  - [ ] Kritische Funktionen implementieren
  - [ ] Unwichtige entfernen oder als deprecated markieren

## ⚠️ WARNUNGEN UND VERSTECKTE PROBLEME

### 11. Massive Warning-Unterdrückung
- **Problem**: Hunderte von Warnings werden unterdrückt
- **Details**:
  - `src/astro_lab/utils/bpy/core.py` Zeile 16-24: Alle NumPy Warnings unterdrückt
  - `test/conftest.py` Zeile 27-29: Torch und Deprecation Warnings unterdrückt
  - Überall: `warnings.filterwarnings("ignore")` versteckt echte Probleme
- **Fix**:
  - [ ] Warning-Unterdrückung reduzieren
  - [ ] Nur spezifische, bekannte Warnings unterdrücken
  - [ ] Echte Probleme beheben statt verstecken

### 12. Memory Leaks und Resource-Probleme
- **Problem**: Komplexe Memory-Management-Logik die nicht funktioniert
- **Details**:
  - `src/astro_lab/cli/__main__.py` Zeile 441: "Memory context: 541948 objects not freed"
  - Zeile 106-116: Komplexe Cleanup-Logik die versagt
  - CUDA Memory wird nicht richtig freigeräben
- **Fix**:
  - [ ] Memory-Management vereinfachen
  - [ ] CUDA Cleanup verbessern
  - [ ] Memory Leaks finden und beheben

### 13. Hardcoded astronomische Konstanten
- **Problem**: Astronomische Werte sind hardcoded ohne Quellen
- **Details**:
  - `src/astro_lab/data/preprocessing.py` Zeile 982, 987: "radius": 180.0 - Was ist das?
  - `src/astro_lab/data/core.py` Zeile 170: `(4 / 3) * np.pi * (radius_pc**3)` - OK, aber keine Kommentare
  - Überall: Magic Numbers ohne Erklärung
- **Fix**:
  - [ ] Astronomische Konstanten in separate Datei
  - [ ] Quellen und Einheiten dokumentieren
  - [ ] Magic Numbers durch benannte Konstanten ersetzen

### 14. Inconsistent Error Handling
- **Problem**: Fehlerbehandlung ist überall unterschiedlich
- **Details**:
  - Manchmal ValueError, manchmal RuntimeError, manchmal logging.error
  - `src/astro_lab/datasets/astro_dataset.py` Zeile 116-123: Verschiedene ValueError für ähnliche Probleme
  - Keine einheitliche Error-Hierarchie
- **Fix**:
  - [ ] Einheitliche Error-Klassen definieren
  - [ ] Konsistente Fehlerbehandlung
  - [ ] Bessere Error-Messages mit Kontext

## 📁 DATEIEN DIE ICH KAPUTT GEMACHT HABE

### Komplett kaputt:
- `src/astro_lab/cli/__init__.py` - Import-Chaos
- `src/astro_lab/cli/__main__.py` - Entry Point Probleme  
- `src/astro_lab/data/datamodule.py` - Unnötige Auto-Setup Logik
- `src/astro_lab/data/preprocessing.py` - Doppelungen und Redundanz
- `src/astro_lab/datasets/astro_dataset.py` - Linter-Warnungen
- `pyproject.toml` - Entry Point zeigt auf kaputte Datei
- `.venv/` - Komplette Environment korrupt

### Funktioniert noch aber hat Probleme:
- `src/astro_lab/tensors/spatial_3d.py` - Astronomische Ungenauigkeiten
- `src/astro_lab/tensors/statistics.py` - Magnitude-Berechnungen fehlerhaft
- `src/astro_lab/data/core.py` - NotImplementedError und Memory Issues

### Versteckte Probleme:
- Alle `src/astro_lab/utils/bpy/` Dateien - Warning-Unterdrückung
- Alle Test-Dateien - Warnings unterdrückt
- `src/astro_lab/__init__.py` - NumPy Warnings versteckt

## 📋 REPARATUR-REIHENFOLGE (PRIORITÄT)

### Phase 1: Kritische Reparaturen (SOFORT)
1. [🟡] Virtual Environment fast repariert - nur noch kleinere Probleme
2. [ ] CLI Import-Probleme beheben
3. [ ] Entry Point korrigieren
4. [ ] DataModule Auto-Setup entfernen

### Phase 2: Mathematische Korrektheit
5. [ ] Koordinaten-Transformationen korrigieren
6. [ ] NaN-Handling vereinheitlichen
7. [ ] Magnitude-Berechnungen korrigieren
8. [ ] Distanz-Berechnungen validieren

### Phase 3: Architektur-Cleanup
9. [ ] Preprocessing Redundanz entfernen
10. [ ] Dataset Type-Probleme beheben
11. [ ] NotImplementedError bewerten und beheben

### Phase 4: Qualität und Stabilität
12. [ ] Warning-Unterdrückung reduzieren
13. [ ] Memory Leaks beheben
14. [ ] Astronomische Konstanten dokumentieren
15. [ ] Error Handling vereinheitlichen

### Phase 5: Testing und Validation
16. [ ] Alle mathematischen Funktionen testen
17. [ ] Astronomische Korrektheit validieren
18. [ ] Performance-Tests
19. [ ] Memory-Tests

## 🎯 ZIEL

**Das System soll wieder funktionieren UND astronomisch korrekt sein:**
- Funktionierendes CLI und Training
- Mathematisch korrekte Koordinaten-Transformationen
- Sinnvolle NaN-Behandlung
- Astronomisch valide Magnitude-Berechnungen
- Keine versteckten Warnings
- Saubere Architektur

**Was ich NICHT mehr machen werde:**
- Warnings unterdrücken statt Probleme zu lösen
- Mathematische Werte mit 0 ersetzen
- Neue Funktionen ohne Tests hinzufügen
- Astronomische Formeln ohne Validierung verwenden

## ⏰ DEADLINE: MORGEN ABGABE

**Priorität: Funktionsfähigkeit UND Korrektheit!**

## 📊 STATISTIK DER PROBLEME

- **Kritische Probleme**: 3
- **Mathematische/Astronomische Fehler**: 4  
- **Architektur-Probleme**: 4
- **Versteckte Probleme**: 4
- **Kaputte Dateien**: 7
- **Problematische Dateien**: 10+
- **Geschätzte Reparatur-Zeit**: 4-6 Stunden

**Das ist ein Desaster. Aber systematisch reparierbar.** 