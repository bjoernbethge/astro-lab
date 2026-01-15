# TensorDict Integration Analysis für AstroLab

## 🔍 Aktuelle Situation

Nach Analyse der aktuellen TensorDict API (v0.8.3) und unserer Implementierung habe ich folgende Erkenntnisse:

### ❌ **Probleme mit unserer aktuellen Integration:**

1. **Wir nutzen TensorDict als einfachen Dict-Wrapper**
   - Unsere `AstroTensorDict` erweitert TensorDict nur um Metadaten
   - Keine Nutzung der Core-Features wie `consolidate()`, `memmap()`, `lazy_stack()`
   - Keine Performance-Vorteile gegenüber normalen Dictionaries

2. **Fehlende TensorDictModule Integration**
   - Wir haben zwar `TensorDictModule` in einigen Encodern
   - Aber die Models selbst arbeiten nicht mit TensorDict als Input/Output
   - Training Loop nutzt PyG Data objects, nicht TensorDict

3. **Verpasste Performance-Features**
   - Kein asynchroner Device Transfer
   - Keine Memory-mapped Tensors für große Datasets
   - Kein `torch.compile` Support
   - Keine Nutzung von `consolidate()` für schnelle Inter-Node Communication

4. **Keine echte Modularität**
   - Models erwarten spezifische PyG Data Struktur
   - Keine wiederverwendbaren TensorDictModule Komponenten
   - Loss Functions arbeiten nicht mit TensorDict

## 📋 Konkrete TODO Liste

### 1. **Core