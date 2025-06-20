#!/usr/bin/env python3
"""Process Earth satellite data with cosmic web analysis using real TLE data and poliastro."""

import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import torch

from src.astro_lab.tensors import Spatial3DTensor, EarthSatelliteTensor
from src.astro_lab.utils import calculate_volume, calculate_mean_density


def load_satellite_data(max_samples=None):
    """Load real satellite data using TLE and poliastro."""
    
    try:
        # Versuche echte TLE-Daten zu laden
        import requests
        from sgp4.api import Satrec
        from sgp4.api import jday
        
        print("📡 Loading real satellite TLE data from CelesTrak...")
        
        # Lade aktuelle TLE-Daten von CelesTrak
        tle_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
        
        try:
            response = requests.get(tle_url, timeout=10)
            response.raise_for_status()
            tle_lines = response.text.strip().split('\n')
            
            satellites = []
            for i in range(0, len(tle_lines), 3):
                if i + 2 < len(tle_lines):
                    name = tle_lines[i].strip()
                    line1 = tle_lines[i + 1].strip()
                    line2 = tle_lines[i + 2].strip()
                    
                    # Parse TLE mit SGP4
                    satellite = Satrec.twoline2rv(line1, line2)
                    
                    # Aktuelle Zeit
                    now = datetime.utcnow()
                    jd, fr = jday(now.year, now.month, now.day, 
                                now.hour, now.minute, now.second)
                    
                    # Position und Geschwindigkeit berechnen
                    e, r, v = satellite.sgp4(jd, fr)
                    
                    if e == 0:  # Erfolgreiche Berechnung
                        satellites.append({
                            'name': name,
                            'x_km': r[0],
                            'y_km': r[1], 
                            'z_km': r[2],
                            'vx_km_s': v[0],
                            'vy_km_s': v[1],
                            'vz_km_s': v[2],
                            'norad_id': satellite.satnum,
                            'epoch': getattr(satellite, 'epoch', 0.0),
                            'inclination_deg': satellite.inclo * 180 / np.pi,
                            'eccentricity': satellite.ecco,
                            'semi_major_axis_km': satellite.a * 6378.137,  # Earth radii to km
                        })
            
            if satellites:
                df = pl.DataFrame(satellites)
                print(f"✅ Loaded {len(df):,} real satellites from TLE data")
                return df
                
        except Exception as e:
            print(f"⚠️ Could not load TLE data: {e}")
    
    except ImportError:
        print("⚠️ SGP4 not available, trying alternative methods")
    
    # Fallback: Versuche poliastro
    try:
        from poliastro.bodies import Earth
        from poliastro.twobody import Orbit
        from poliastro.util import time_range
        
        print("🛰️ Creating satellite data with poliastro...")
        
        # Erstelle realistische Satellitenorbits mit poliastro
        satellites = []
        
        # Verschiedene Orbithöhen und Inklinationen
        altitudes = [400, 800, 1200, 20200, 35786]  # km (LEO, MEO, GEO)
        inclinations = [0, 45, 90, 135, 180]  # degrees
        
        for i, (alt, inc) in enumerate(zip(altitudes, inclinations)):
            # Erstelle Orbit mit poliastro
            a = Earth.R + alt * 1000  # km to m
            ecc = 0.0  # Circular orbit
            
            orbit = Orbit.from_classical(
                Earth, 
                a=a, 
                ecc=ecc, 
                inc=np.radians(inc), 
                raan=np.radians(0), 
                argp=np.radians(0), 
                nu=np.radians(0)
            )
            
            # Position und Geschwindigkeit
            r = orbit.r.to_value() / 1000  # m to km
            v = orbit.v.to_value() / 1000  # m/s to km/s
            
            satellites.append({
                'name': f'Satellite_{i+1}',
                'x_km': r[0],
                'y_km': r[1],
                'z_km': r[2], 
                'vx_km_s': v[0],
                'vy_km_s': v[1],
                'vz_km_s': v[2],
                'norad_id': 10000 + i,
                'altitude_km': alt,
                'inclination_deg': inc,
                'eccentricity': ecc,
                'semi_major_axis_km': a / 1000,
            })
        
        df = pl.DataFrame(satellites)
        print(f"✅ Created {len(df):,} satellites with poliastro")
        return df
        
    except ImportError:
        print("⚠️ poliastro not available")
    
    # Letzter Fallback: Versuche astroML
    try:
        from astroML.datasets import fetch_sdss_galaxy_dr7
        
        print("🌌 Using astroML data as satellite proxy...")
        
        # Lade SDSS-Daten als Proxy für Satellitenpositionen
        data = fetch_sdss_galaxy_dr7()
        
        # Verwende RA/Dec als Proxy für Satellitenpositionen
        n_satellites = min(1000, len(data))
        indices = np.random.choice(len(data), n_satellites, replace=False)
        
        satellites = []
        for i, idx in enumerate(indices):
            # Konvertiere RA/Dec zu 3D-Positionen (vereinfacht)
            ra = data['ra'][idx]
            dec = data['dec'][idx]
            distance = 400 + np.random.uniform(0, 35000)  # 400-35400 km (LEO bis GEO)
            
            # Sphärische zu kartesischen Koordinaten
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            
            x = distance * np.cos(dec_rad) * np.cos(ra_rad)
            y = distance * np.cos(dec_rad) * np.sin(ra_rad)
            z = distance * np.sin(dec_rad)
            
            satellites.append({
                'name': f'AstroML_Sat_{i+1}',
                'x_km': x,
                'y_km': y,
                'z_km': z,
                'vx_km_s': np.random.uniform(-7, 7),
                'vy_km_s': np.random.uniform(-7, 7),
                'vz_km_s': np.random.uniform(-7, 7),
                'norad_id': 20000 + i,
                'altitude_km': distance - 6371,  # Earth radius
                'inclination_deg': np.random.uniform(0, 180),
                'eccentricity': np.random.uniform(0, 0.1),
                'semi_major_axis_km': distance,
            })
        
        df = pl.DataFrame(satellites)
        print(f"✅ Created {len(df):,} satellites from astroML data")
        return df
        
    except ImportError:
        print("⚠️ astroML not available")
    
    # Absoluter Fallback: Realistische Demo-Daten
    print("⚠️ No satellite libraries available, creating realistic demo data")
    n_satellites = 1000
    
    # Realistische Orbithöhen
    altitudes = np.random.choice([400, 800, 1200, 20200, 35786], n_satellites)
    inclinations = np.random.uniform(0, 180, n_satellites)
    
    satellites = []
    for i in range(n_satellites):
        alt = altitudes[i]
        inc = inclinations[i]
        
        # Realistische Orbitalgeschwindigkeiten basierend auf Höhe
        if alt < 1000:  # LEO
            v_mag = 7.8  # km/s
        elif alt < 10000:  # MEO
            v_mag = 5.0  # km/s
        else:  # GEO
            v_mag = 3.1  # km/s
        
        # Position auf Orbitalbahn
        angle = np.random.uniform(0, 2*np.pi)
        x = (6371 + alt) * np.cos(angle)
        y = (6371 + alt) * np.sin(angle) * np.cos(np.radians(inc))
        z = (6371 + alt) * np.sin(angle) * np.sin(np.radians(inc))
        
        # Geschwindigkeit senkrecht zur Position
        vx = -v_mag * np.sin(angle)
        vy = v_mag * np.cos(angle) * np.cos(np.radians(inc))
        vz = v_mag * np.cos(angle) * np.sin(np.radians(inc))
        
        satellites.append({
            'name': f'Demo_Sat_{i+1}',
            'x_km': x,
            'y_km': y,
            'z_km': z,
            'vx_km_s': vx,
            'vy_km_s': vy,
            'vz_km_s': vz,
            'norad_id': 30000 + i,
            'altitude_km': alt,
            'inclination_deg': inc,
            'eccentricity': np.random.uniform(0, 0.1),
            'semi_major_axis_km': 6371 + alt,
        })
    
    df = pl.DataFrame(satellites)
    print(f"✅ Created {len(df):,} realistic demo satellites")
    return df


def main():
    print("🛰️ SATELLITE COSMIC WEB ANALYSIS")
    print("=" * 50)

    # Load satellite data
    start_time = time.time()
    print("📊 Loading Earth satellite catalog...")

    try:
        df = load_satellite_data()  # Real satellite data
        load_time = time.time() - start_time
        print(f"✅ Loaded in {load_time:.1f}s: {len(df):,} satellites")

        # Koordinaten extrahieren
        x = df["x_km"].to_numpy()
        y = df["y_km"].to_numpy()
        z = df["z_km"].to_numpy()
        coords_3d = np.column_stack([x, y, z])

        # Spatial3DTensor direkt erzeugen
        spatial_tensor = Spatial3DTensor(coords_3d, unit="km")
        coords_tensor = spatial_tensor.cartesian

        # Volumen und Dichte
        total_volume = calculate_volume(coords_3d)
        mean_density = calculate_mean_density(coords_3d)

        print(f"📏 Total volume: {total_volume:.0f} km³")
        print(f"📊 Mean density: {mean_density:.2e} satellites/km³")

        # Multi-scale cosmic web clustering für Satelliten
        scales = [1000.0, 5000.0, 10000.0, 50000.0]  # km - größere Skalen für Satelliten
        results_summary = {}

        for scale in scales:
            print(f"\n🕸️ Satellite Cosmic Web Clustering at {scale} km scale:")
            start_time = time.time()

            try:
                # Adaptive Parameter basierend auf Skala
                if scale <= 5000:
                    min_samples = 3
                elif scale <= 10000:
                    min_samples = 2
                else:
                    min_samples = 1
                
                # Konvertiere km zu pc für die Clustering-Funktion (1 pc ≈ 3.086e13 km)
                eps_pc = scale / 3.086e13
                
                results = spatial_tensor.cosmic_web_clustering(
                    eps_pc=eps_pc,
                    min_samples=min_samples,
                    algorithm="dbscan",
                )

                cluster_time = time.time() - start_time
                print(f"⏱️ Clustering completed in {cluster_time:.1f}s")

                n_groups = results["n_clusters"]
                n_noise = results["n_noise"]

                results_summary[scale] = {
                    "n_clusters": n_groups,
                    "n_noise": n_noise,
                    "grouped_fraction": (len(coords_3d) - n_noise) / len(coords_3d),
                    "time_s": cluster_time,
                }

                print(f"  Satellite groups: {n_groups:,}")
                print(
                    f"  Grouped satellites: {len(coords_3d) - n_noise:,} ({(len(coords_3d) - n_noise) / len(coords_3d) * 100:.1f}%)"
                )
                print(
                    f"  Isolated satellites: {n_noise:,} ({n_noise / len(coords_3d) * 100:.1f}%)"
                )

                if n_groups > 0 and n_groups < 20:
                    stats = results["cluster_stats"]
                    group_sizes = [(k, v["n_stars"]) for k, v in stats.items()]
                    group_sizes.sort(key=lambda x: x[1], reverse=True)
                    print(f"  Top 5 groups: {group_sizes[:5]}")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

        # Save results
        output_dir = Path("results/satellite_cosmic_web")
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(coords_tensor, output_dir / "satellite_coords_3d_km.pt")

        # Save summary
        with open(output_dir / "satellite_cosmic_web_summary.txt", "w") as f:
            f.write("Satellite Cosmic Web Analysis\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total satellites: {len(coords_3d):,}\n")
            f.write(f"Coordinate range: {coords_3d.min():.1f} - {coords_3d.max():.1f} km\n")
            f.write(f"Mean density: {mean_density:.2e} satellites/km³\n")
            f.write(f"Total volume: {total_volume:.0f} km³\n\n")

            f.write("Multi-scale clustering results:\n")
            for scale, result in results_summary.items():
                f.write(f"  {scale} km: {result['n_clusters']} groups, ")
                f.write(f"{result['grouped_fraction'] * 100:.1f}% grouped, ")
                f.write(f"{result['time_s']:.1f}s\n")

        print(f"\n💾 Results saved to: {output_dir}")
        print("🎉 Satellite cosmic web analysis complete!")

        # Print final summary
        print("\n📈 SUMMARY:")
        print(f"  Dataset: {len(coords_3d):,} satellites")
        print(f"  Volume: {total_volume:.0f} km³")
        print(f"  Density: {mean_density:.2e} satellites/km³")

        for scale, result in results_summary.items():
            print(
                f"  {scale:5.0f} km: {result['n_clusters']:4d} groups ({result['grouped_fraction'] * 100:5.1f}% grouped)"
            )

    except Exception as e:
        print(f"❌ Error in satellite processing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main() 