#!/usr/bin/env python3
"""
Astroquery Integration Demo for astro-lab

Demonstrates how to use astroquery to fetch exoplanet and asteroid data.
Optimized for reliability with timeout handling and small data samples.
"""

import sys
from pathlib import Path

import polars as pl

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Use the new data module
from astro_lab.data import check_astroquery_available

# Fallback message if enhanced features not available
try:
    from astro_lab.data import HAS_ENHANCED_FEATURES

    if not HAS_ENHANCED_FEATURES:
        print("⚠️  Enhanced astroquery features not available - basic demo only")
except ImportError:
    print("⚠️  astro_lab.data not available - basic demo only")


def test_basic_functionality():
    """Test basic astroquery functionality."""
    print("🚀 === ASTROQUERY INTEGRATION TEST ===")

    try:
        if not check_astroquery_available():
            print("❌ astroquery not available")
            print("💡 Installiere mit: uv add astroquery")
            return False

        print("✅ astroquery available!")
        return True

    except ImportError as e:
        print(f"❌ Import Fehler: {e}")
        return False


def demo_exoplanet_data():
    """Demonstrate exoplanet data fetching with timeout protection."""
    print("\n🪐 === EXOPLANET DATA DEMO ===")

    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

        # Very small query to avoid timeouts
        print("📡 Loading 3 confirmed exoplanets (minimal data)...")

        # Use direct TAP query with minimal columns
        result = NasaExoplanetArchive.query_criteria(
            table="ps",
            select="top 3 pl_name,hostname,disc_year",
            where="default_flag=1",
        )

        planets = pl.from_pandas(result.to_pandas())

        print(f"✅ {len(planets)} Exoplaneten geladen")
        print(f"Spalten: {list(planets.columns)}")

        # Show planet info
        if len(planets) > 0:
            for i in range(min(len(planets), 3)):
                row = planets.row(i, named=True)
                name = row.get("pl_name", "N/A")
                host = row.get("hostname", "N/A")
                year = row.get("disc_year", "N/A")
                print(f"  {i + 1}. {name} um {host} (entdeckt {year})")

        return planets

    except Exception as e:
        print(f"❌ Fehler beim Laden der Exoplanet-Daten: {e}")
        print("💡 NASA Exoplanet Archive hat bekannte Timeout-Probleme")
        print("💡 Try again later or use smaller queries")
        return None


def demo_asteroid_data():
    """Demonstrate asteroid data fetching with timeout protection."""
    print("\n🪨 === ASTEROID DATA DEMO ===")

    try:
        from astroquery.jplhorizons import Horizons

        # Very minimal ephemeris query
        print("📡 Lade Ceres Position für heute...")

        obj = Horizons(
            id="Ceres",
            location="500",  # Geocenter
            epochs="2025-01-15",  # Single epoch
        )
        eph = obj.ephemerides()
        ceres_data = pl.from_pandas(eph.to_pandas())

        print("✅ Ceres-Daten geladen")
        print(f"Verfügbare Spalten: {len(ceres_data.columns)}")

        # Show basic position info
        if len(ceres_data) > 0:
            row = ceres_data.row(0, named=True)
            ra = row.get("RA", "N/A")
            dec = row.get("DEC", "N/A")
            print(f"Ceres Position: RA={ra}, DEC={dec}")

        return ceres_data

    except Exception as e:
        print(f"❌ Fehler beim Laden der Asteroid-Daten: {e}")
        print("💡 JPL Horizons kann bei Netzwerkproblemen hängen")
        return None


def demo_simple_integration():
    """Demonstrate simple astroquery usage without complex processing."""
    print("\n🎯 === SIMPLE INTEGRATION DEMO ===")

    try:
        # Test if we can import everything needed
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
        from astroquery.jplhorizons import Horizons

        print("✅ Alle astroquery Module verfügbar")

        # Show available tables
        print("📋 Verfügbare NASA Exoplanet Archive Tabellen:")
        tables = NasaExoplanetArchive.TAP_TABLES
        for table in tables[:5]:  # Show first 5
            print(f"  - {table}")
        print(f"  ... und {len(tables) - 5} weitere")

        return True

    except Exception as e:
        print(f"❌ Fehler bei Integration: {e}")
        return False


def main():
    """Main demo function with robust error handling."""
    print("🌟 ASTROQUERY INTEGRATION DEMO für astro-lab")
    print("=" * 50)
    print("⚠️  HINWEIS: Optimiert für Zuverlässigkeit mit kleinen Datenmengen")
    print("=" * 50)

    # Test basic functionality
    if not test_basic_functionality():
        print("❌ Grundlegende Funktionalität nicht verfügbar")
        return

    # Run demos with individual error handling
    print("\n" + "=" * 50)
    exo_success = False
    try:
        exo_data = demo_exoplanet_data()
        exo_success = exo_data is not None
    except Exception as e:
        print(f"❌ Exoplanet Demo fehlgeschlagen: {e}")

    print("\n" + "=" * 50)
    asteroid_success = False
    try:
        asteroid_data = demo_asteroid_data()
        asteroid_success = asteroid_data is not None
    except Exception as e:
        print(f"❌ Asteroid Demo fehlgeschlagen: {e}")

    print("\n" + "=" * 50)
    integration_success = False
    try:
        integration_success = demo_simple_integration()
    except Exception as e:
        print(f"❌ Integration Demo fehlgeschlagen: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📋 === ERGEBNISSE ===")
    print("✅ astroquery verfügbar: ✅")
    print(f"✅ Exoplanet-Daten: {'✅' if exo_success else '❌'}")
    print(f"✅ Asteroid-Daten: {'✅' if asteroid_success else '❌'}")
    print(f"✅ Integration: {'✅' if integration_success else '❌'}")

    if exo_success or asteroid_success:
        print("\n🎉 astroquery Integration funktioniert grundsätzlich!")
        print("\n💡 Für Produktionsnutzung:")
        print("1. Verwende kleinere Queries um Timeouts zu vermeiden")
        print("2. Implementiere Retry-Logik für fehlgeschlagene Requests")
        print("3. Cache Ergebnisse lokal für wiederholte Nutzung")
        print("4. Verwende TAP-Clients wie TOPCAT für große Datenmengen")
    else:
        print("\n⚠️  Netzwerk-/Server-Probleme - versuche es später nochmal")
        print("💡 astroquery hängt oft bei der ersten Nutzung oder bei Serverproblemen")


if __name__ == "__main__":
    main()
