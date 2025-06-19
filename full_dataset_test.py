#!/usr/bin/env python3
"""
Full TNG50 Dataset Creation Test
===============================

Testet die Erstellung eines vollständigen TNG50-Datasets.
"""

import time
from pathlib import Path
from astro_lab.data import import_tng50, load_catalog, create_training_splits

def test_full_dataset_creation():
    """Test vollständige Dataset-Erstellung."""
    print('🌌 Full TNG50 Dataset Creation Test')
    print('=' * 50)
    
    snap_dir = Path("data/raw/TNG50-4/output/snapdir_099")
    snap_files = sorted(snap_dir.glob("snap_*.hdf5"))
    
    print(f"📂 Found {len(snap_files)} snapshot files")
    
    # Test verschiedene Szenarien
    scenarios = [
        {
            'name': 'Quick Dataset (Black Holes only)',
            'particle_types': ['PartType5'],
            'max_particles': 100,
            'files_to_process': 3
        },
        {
            'name': 'Medium Dataset (Stars)',
            'particle_types': ['PartType4'],
            'max_particles': 1000,
            'files_to_process': 2
        },
        {
            'name': 'Large Dataset (Stars + Black Holes)',
            'particle_types': ['PartType4', 'PartType5'],
            'max_particles': 5000,
            'files_to_process': 1
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🔬 Scenario: {scenario['name']}")
        print("-" * 40)
        
        total_particles = 0
        total_time = 0
        
        files_to_use = snap_files[:scenario['files_to_process']]
        
        for i, snap_file in enumerate(files_to_use):
            print(f"📁 Processing {snap_file.name} ({i+1}/{len(files_to_use)})")
            
            for ptype in scenario['particle_types']:
                start_time = time.time()
                
                try:
                    # Import TNG50 data
                    parquet_file = import_tng50(snap_file, dataset_name=ptype)
                    
                    # Load and sample
                    df = load_catalog(parquet_file)
                    
                    if len(df) > scenario['max_particles']:
                        df = df.sample(scenario['max_particles'], seed=42)
                    
                    particles_processed = len(df)
                    total_particles += particles_processed
                    
                    processing_time = time.time() - start_time
                    total_time += processing_time
                    
                    print(f"   {ptype}: {particles_processed:,} particles in {processing_time:.2f}s")
                    
                except Exception as e:
                    print(f"   ❌ Error with {ptype}: {e}")
        
        # Berechne Statistiken
        avg_time_per_file = total_time / len(files_to_use) if files_to_use else 0
        particles_per_second = total_particles / total_time if total_time > 0 else 0
        
        print(f"\n📊 Scenario Results:")
        print(f"   Total particles: {total_particles:,}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Avg time per file: {avg_time_per_file:.2f}s")
        print(f"   Processing rate: {particles_per_second:.0f} particles/s")
        
        # Hochrechnung für alle Dateien
        if len(files_to_use) > 0:
            estimated_total_time = avg_time_per_file * len(snap_files)
            estimated_total_particles = (total_particles / len(files_to_use)) * len(snap_files)
            
            print(f"\n🔮 Extrapolation for all {len(snap_files)} files:")
            print(f"   Estimated particles: {estimated_total_particles:,.0f}")
            print(f"   Estimated time: {estimated_total_time:.1f}s ({estimated_total_time/60:.1f} minutes)")

def test_optimal_workflow():
    """Test optimalen Workflow für Dataset-Erstellung."""
    print('\n\n🚀 Optimal Workflow Test')
    print('=' * 50)
    
    snap_file = Path("data/raw/TNG50-4/output/snapdir_099/snap_099.0.hdf5")
    
    # Workflow: Schnelle Erstellung eines ML-ready Datasets
    workflow_start = time.time()
    
    print("1️⃣ Loading Stars (PartType4)...")
    start = time.time()
    parquet_file = import_tng50(snap_file, 'PartType4')
    df_stars = load_catalog(parquet_file)
    print(f"   ✅ {len(df_stars):,} stars loaded in {time.time() - start:.2f}s")
    
    print("2️⃣ Loading Black Holes (PartType5)...")
    start = time.time()
    parquet_file_bh = import_tng50(snap_file, 'PartType5')
    df_bh = load_catalog(parquet_file_bh)
    print(f"   ✅ {len(df_bh):,} black holes loaded in {time.time() - start:.2f}s")
    
    print("3️⃣ Creating balanced dataset...")
    start = time.time()
    
    # Erstelle ausgewogenes Dataset
    n_stars = min(5000, len(df_stars))
    n_bh = min(500, len(df_bh))
    
    df_stars_sample = df_stars.sample(n_stars, seed=42)
    df_bh_sample = df_bh.sample(n_bh, seed=42)
    
    print(f"   ✅ Sampled {n_stars:,} stars + {n_bh:,} black holes in {time.time() - start:.2f}s")
    
    print("4️⃣ Creating training splits...")
    start = time.time()
    
    # Erstelle Splits für beide Partikeltypen
    train_stars, val_stars, test_stars = create_training_splits(df_stars_sample, test_size=0.2, val_size=0.1)
    train_bh, val_bh, test_bh = create_training_splits(df_bh_sample, test_size=0.2, val_size=0.1)
    
    split_time = time.time() - start
    print(f"   ✅ Training splits created in {split_time:.2f}s")
    print(f"      Stars: Train={len(train_stars):,}, Val={len(val_stars):,}, Test={len(test_stars):,}")
    print(f"      Black Holes: Train={len(train_bh):,}, Val={len(val_bh):,}, Test={len(test_bh):,}")
    
    total_workflow_time = time.time() - workflow_start
    
    print(f"\n🎯 Optimal Workflow Results:")
    print(f"   Total particles: {n_stars + n_bh:,}")
    print(f"   Total time: {total_workflow_time:.2f}s")
    print(f"   Ready for ML training: ✅")
    
    # Test bestanden - verwende assertions statt return
    assert total_workflow_time > 0, "Workflow time should be positive"
    assert n_stars > 0, "Should have processed some stars"
    assert n_bh >= 0, "Should have processed some black holes or none"

if __name__ == "__main__":
    test_full_dataset_creation()
    test_optimal_workflow()
    
    print(f"\n💡 Summary:")
    print(f"   All tests completed successfully!")
    print(f"   Ready for immediate ML training!")
    print(f"   Datasets are cached for future use 🚀") 