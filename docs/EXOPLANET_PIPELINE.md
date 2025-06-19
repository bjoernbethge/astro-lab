# Exoplanet-Daten: Was geht, was nicht

## ❌ Aktueller Status

**Exoplanet-Funktionalität ist NICHT in `astro_lab.data` verfügbar.** Das Hauptpaket fokussiert sich auf Gaia, SDSS, NSA und LINEAR.

## ✅ Was du stattdessen machen kannst

### Option 1: Direkt mit astroquery (30 Sekunden)

```python
# NASA Exoplanet Archive direkt abfragen
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import polars as pl

try:
    # Kleine Query (funktioniert meist)
    result = NasaExoplanetArchive.query_criteria(
        table="ps",
        select="top 100 pl_name,hostname,pl_rade,pl_masse,sy_dist,disc_year",
        where="default_flag=1"
    )
    
    exoplanets = pl.from_pandas(result.to_pandas())
    print(f"✅ {len(exoplanets)} Exoplaneten geladen")
    
except Exception as e:
    print(f"❌ NASA Archive down: {e}")
    print("💡 Verwende Demo-Daten (siehe unten)")
```

### Option 2: Demo-Daten generieren (immer funktioniert)

```python
import numpy as np
import polars as pl

# Realistische Exoplanet-Demo-Daten
n_planets = 1000

demo_exoplanets = pl.DataFrame({
    'pl_name': [f'Demo-{i}b' for i in range(n_planets)],
    'hostname': [f'Demo-{i}' for i in range(n_planets)],
    'ra': np.random.uniform(0, 360, n_planets),
    'dec': np.random.uniform(-90, 90, n_planets),
    'sy_dist': np.random.lognormal(2, 1, n_planets),  # parsecs
    'pl_rade': np.random.lognormal(0, 0.5, n_planets),  # Earth radii
    'pl_masse': np.random.lognormal(0, 1, n_planets),   # Earth masses
    'disc_year': np.random.randint(1995, 2024, n_planets)
})

print(f"✅ {len(demo_exoplanets)} Demo-Exoplaneten erstellt")
```

### Option 3: astro-torch Package (für Graphs)

```python
# Separates Package installieren: pip install astro-torch
try:
    from astro_torch.data import ExoplanetGraphDataset
    
    dataset = ExoplanetGraphDataset(max_planets=1000)
    graph = dataset[0]
    print(f"Graph: {graph.num_nodes} Knoten, {graph.num_edges} Kanten")
    
except ImportError:
    print("❌ astro-torch nicht installiert")
    print("💡 Verwende Option 1 oder 2")
```

## 🚀 Praktische Exoplanet-Analyse

### Rezept 1: Planet-Größen analysieren

```python
# 1. Daten laden (Demo oder echt)
exoplanets = demo_exoplanets  # Oder von astroquery

# 2. Größen-Verteilung
import matplotlib.pyplot as plt

radii = exoplanets['pl_rade'].to_numpy()
radii = radii[~np.isnan(radii)]  # NaN entfernen

# 3. Kategorien
earth_like = np.sum(radii < 1.5)
super_earth = np.sum((radii >= 1.5) & (radii < 4))
neptune_like = np.sum((radii >= 4) & (radii < 10))
jupiter_like = np.sum(radii >= 10)

print(f"🌍 Earth-like: {earth_like}")
print(f"🌎 Super-Earth: {super_earth}")
print(f"🔵 Neptune-like: {neptune_like}")
print(f"🪐 Jupiter-like: {jupiter_like}")

# 4. Plot
plt.hist(radii, bins=50, alpha=0.7)
plt.xlabel('Planet Radius (Earth radii)')
plt.ylabel('Count')
plt.title('Exoplanet Size Distribution')
plt.show()
```

### Rezept 2: Discovery Timeline

```python
# Entdeckungen pro Jahr
years = exoplanets['disc_year'].to_numpy()
years = years[~np.isnan(years)]

# Zählen
from collections import Counter
year_counts = Counter(years)

# Plot
years_sorted = sorted(year_counts.keys())
counts = [year_counts[year] for year in years_sorted]

plt.plot(years_sorted, counts, 'o-')
plt.xlabel('Discovery Year')
plt.ylabel('Number of Planets')
plt.title('Exoplanet Discoveries Over Time')
plt.grid(True)
plt.show()

print(f"Peak year: {years_sorted[np.argmax(counts)]} ({max(counts)} planets)")
```

### Rezept 3: Host Star Distance

```python
# Distanz-Analyse
distances = exoplanets['sy_dist'].to_numpy()
distances = distances[~np.isnan(distances)]

print(f"Closest system: {np.min(distances):.1f} parsecs")
print(f"Farthest system: {np.max(distances):.1f} parsecs")
print(f"Median distance: {np.median(distances):.1f} parsecs")

# Nearby systems (< 50 parsecs)
nearby = np.sum(distances < 50)
print(f"🏠 Nearby systems (<50 pc): {nearby}")

# Plot
plt.hist(distances, bins=50, alpha=0.7)
plt.axvline(50, color='red', linestyle='--', label='50 parsecs')
plt.xlabel('Distance (parsecs)')
plt.ylabel('Count')
plt.legend()
plt.title('Host Star Distances')
plt.show()
```

## 🔧 Häufige Probleme & Lösungen

### Problem: NASA Archive Timeout
```python
# Lösung: Kleinere Queries + Retry
def safe_nasa_query(max_planets=100, retries=3):
    for i in range(retries):
        try:
            result = NasaExoplanetArchive.query_criteria(
                table="ps",
                select=f"top {max_planets} pl_name,pl_rade,pl_masse",
                where="default_flag=1"
            )
            return pl.from_pandas(result.to_pandas())
        except:
            print(f"Retry {i+1}/{retries}...")
            time.sleep(2)
    
    print("❌ NASA Archive nicht erreichbar, verwende Demo-Daten")
    return demo_exoplanets[:max_planets]
```

### Problem: Viele NaN Values
```python
# Lösung: Daten bereinigen
def clean_exoplanet_data(df):
    # Nur Planeten mit Radius und Masse
    df = df.filter(
        pl.col('pl_rade').is_not_null() & 
        pl.col('pl_masse').is_not_null()
    )
    
    # Unrealistische Werte entfernen
    df = df.filter(
        (pl.col('pl_rade') > 0) & (pl.col('pl_rade') < 50) &
        (pl.col('pl_masse') > 0) & (pl.col('pl_masse') < 5000)
    )
    
    return df

clean_data = clean_exoplanet_data(exoplanets)
print(f"Bereinigt: {len(clean_data)} von {len(exoplanets)} Planeten")
```

## 🔮 Zukünftige Integration

### Geplante astro_lab.data Integration

```python
# Das würde in Zukunft funktionieren:
from astro_lab.data import load_exoplanet_data  # NOCH NICHT VERFÜGBAR

# Saubere API wie bei anderen Surveys
exoplanets = load_exoplanet_data(
    max_samples=1000,
    source='nasa_archive',  # oder 'eu_archive'
    return_tensor=True
)

print(f"Exoplanets: {exoplanets.shape}")
```

### Was dafür entwickelt werden muss

1. **Robuste NASA Archive Integration** (Timeouts handhaben)
2. **Caching System** (lokale Datenspeicherung)
3. **Data Quality Filters** (NaN handling, unrealistic values)
4. **SurveyTensor Support** (wie bei Gaia/SDSS)

## 🛠️ Jetzt beitragen

Wenn du Exoplanet-Support implementieren willst:

### Schritt 1: Basis-Integration

```python
# In src/astro_lab/data/core.py erweitern:

SURVEY_CONFIGS['exoplanet'] = {
    'name': 'NASA Exoplanet Archive',
    'coord_cols': ['ra', 'dec', 'sy_dist'],
    'mag_cols': ['pl_rade', 'pl_masse'],
    'extra_cols': ['disc_year', 'hostname'],
    'photometric_bands': ['planet_radius', 'planet_mass'],
    # ... weitere Config
}
```

### Schritt 2: Download Function

```python
def download_exoplanet_data(max_planets=5000):
    """Download exoplanet data with robust error handling."""
    try:
        result = NasaExoplanetArchive.query_criteria(
            table="ps",
            select=f"top {max_planets} *",
            where="default_flag=1"
        )
        return pl.from_pandas(result.to_pandas())
    except Exception as e:
        print(f"NASA Archive error: {e}")
        return generate_demo_exoplanets(max_planets)
```

### Schritt 3: Tests schreiben

```python
def test_exoplanet_loading():
    # Mit Demo-Daten testen (nicht NASA API abhängig)
    data = load_exoplanet_data(max_samples=100)
    assert len(data) == 100
    assert 'pl_rade' in data.columns
```

## 📝 Quick Commands

```bash
# Test NASA Archive Verbindung
uv run python -c "
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
try:
    result = NasaExoplanetArchive.query_criteria(table='ps', select='top 1 pl_name')
    print('✅ NASA Archive erreichbar')
except:
    print('❌ NASA Archive down')
"

# Demo-Daten erstellen
uv run python -c "
import numpy as np
import polars as pl
demo = pl.DataFrame({
    'pl_rade': np.random.lognormal(0, 0.5, 100),
    'pl_masse': np.random.lognormal(0, 1, 100)
})
print(f'✅ {len(demo)} Demo-Planeten erstellt')
"

# Astroquery Demo laufen lassen
uv run python examples/astroquery_demo.py
```

---

**TL;DR**: Exoplanets nicht in astro_lab.data → Verwende astroquery direkt oder Demo-Daten → Beitrag zur Integration willkommen! 🪐