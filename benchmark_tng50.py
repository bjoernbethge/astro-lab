#!/usr/bin/env python3
"""
TNG50 Processing Benchmark
=========================

Testet verschiedene TNG50-Verarbeitungsszenarien und misst Performance.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl

from astro_lab.data import import_tng50, load_catalog, get_data_statistics, create_training_splits
from astro_lab.simulation import TNG50Loader


class TNG50Benchmark:
    """Benchmark für TNG50-Verarbeitung."""
    
    def __init__(self):
        self.snap_file = Path("data/raw/TNG50-4/output/snapdir_099/snap_099.0.hdf5")
        self.results = {}
        
    def time_operation(self, name: str, func, *args, **kwargs):
        """Zeitmessung für eine Operation."""
        print(f"⏱️  Testing: {name}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"   ✅ Completed in {duration:.2f}s")
            self.results[name] = {
                'duration': duration,
                'success': True,
                'result': result
            }
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"   ❌ Failed after {duration:.2f}s: {e}")
            self.results[name] = {
                'duration': duration,
                'success': False,
                'error': str(e)
            }
            return None
    
    def benchmark_particle_types(self):
        """Benchmark verschiedene Partikeltypen."""
        print("\n🌌 Benchmarking Particle Types")
        print("=" * 50)
        
        particle_types = ['PartType4', 'PartType5']  # Stars and Black Holes
        particle_counts = [100, 500, 1000, 5000]
        
        for ptype in particle_types:
            print(f"\n📊 Testing {ptype}:")
            
            for count in particle_counts:
                name = f"{ptype}_{count}_particles"
                
                def load_and_process():
                    # Import TNG50 data
                    parquet_file = import_tng50(self.snap_file, dataset_name=ptype)
                    
                    # Load catalog
                    df = load_catalog(parquet_file)
                    
                    # Sample if needed
                    if len(df) > count:
                        df = df.sample(count, seed=42)
                    
                    # Get statistics
                    stats = get_data_statistics(df)
                    
                    return {
                        'n_particles': len(df),
                        'n_columns': len(df.columns),
                        'memory_mb': stats['memory_usage_mb'],
                        'parquet_file': parquet_file
                    }
                
                result = self.time_operation(name, load_and_process)
                
                if result:
                    print(f"     Particles: {result['n_particles']:,}")
                    print(f"     Columns: {result['n_columns']}")
                    print(f"     Memory: {result['memory_mb']:.1f} MB")
    
    def benchmark_data_splits(self):
        """Benchmark Training-Split-Erstellung."""
        print("\n🔄 Benchmarking Training Splits")
        print("=" * 50)
        
        # Load a medium dataset
        parquet_file = import_tng50(self.snap_file, dataset_name='PartType4')
        df = load_catalog(parquet_file)
        
        # Test different dataset sizes
        sizes = [100, 500, 1000, 5000]
        
        for size in sizes:
            if len(df) >= size:
                df_sample = df.sample(size, seed=42)
                
                name = f"splits_{size}_particles"
                
                def create_splits():
                    train, val, test = create_training_splits(
                        df_sample,
                        test_size=0.2,
                        val_size=0.1,
                        random_state=42
                    )
                    return {
                        'train_size': len(train),
                        'val_size': len(val),
                        'test_size': len(test)
                    }
                
                result = self.time_operation(name, create_splits)
                
                if result:
                    print(f"     Train: {result['train_size']:,}")
                    print(f"     Val: {result['val_size']:,}")
                    print(f"     Test: {result['test_size']:,}")
    
    def benchmark_raw_loading(self):
        """Benchmark Raw HDF5 Loading."""
        print("\n📁 Benchmarking Raw HDF5 Loading")
        print("=" * 50)
        
        particle_limits = [100, 500, 1000, 5000]
        
        for limit in particle_limits:
            name = f"raw_loading_{limit}_particles"
            
            def load_raw():
                data = TNG50Loader.load_snapshot(
                    self.snap_file,
                    particle_types=['PartType4', 'PartType5'],
                    max_particles=limit
                )
                
                total_particles = 0
                for ptype, pdata in data.get('particles', {}).items():
                    coords = pdata.get('coordinates', [])
                    total_particles += len(coords)
                
                return {
                    'total_particles': total_particles,
                    'particle_types': list(data.get('particles', {}).keys()),
                    'header': data.get('header', {})
                }
            
            result = self.time_operation(name, load_raw)
            
            if result:
                print(f"     Total particles: {result['total_particles']:,}")
                print(f"     Particle types: {result['particle_types']}")
    
    def benchmark_file_sizes(self):
        """Benchmark verschiedene Snapshot-Dateien."""
        print("\n📂 Benchmarking Different Snapshot Files")
        print("=" * 50)
        
        snap_dir = Path("data/raw/TNG50-4/output/snapdir_099")
        snap_files = sorted(snap_dir.glob("snap_099.*.hdf5"))[:3]  # Test first 3 files
        
        for snap_file in snap_files:
            size_mb = snap_file.stat().st_size / (1024 * 1024)
            name = f"file_{snap_file.name}_{size_mb:.0f}MB"
            
            def load_file():
                data = TNG50Loader.load_snapshot(
                    snap_file,
                    particle_types=['PartType5'],  # Just Black Holes for speed
                    max_particles=100
                )
                
                bh_data = data.get('particles', {}).get('PartType5', {})
                coords = bh_data.get('coordinates', [])
                
                return {
                    'file_size_mb': size_mb,
                    'black_holes': len(coords),
                    'header': data.get('header', {})
                }
            
            result = self.time_operation(name, load_file)
            
            if result:
                print(f"     File size: {result['file_size_mb']:.0f} MB")
                print(f"     Black holes: {result['black_holes']:,}")
    
    def run_full_benchmark(self):
        """Führe vollständigen Benchmark aus."""
        print("🚀 TNG50 Processing Benchmark")
        print("=" * 60)
        print(f"📁 Snapshot file: {self.snap_file.name}")
        print(f"📊 GPU: NVIDIA GeForce RTX 4070 Laptop GPU")
        print()
        
        # Run all benchmarks
        self.benchmark_raw_loading()
        self.benchmark_particle_types()
        self.benchmark_data_splits()
        self.benchmark_file_sizes()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Drucke Benchmark-Zusammenfassung."""
        print("\n📈 Benchmark Summary")
        print("=" * 60)
        
        # Group results by category
        categories = {
            'Raw Loading': [k for k in self.results.keys() if k.startswith('raw_loading')],
            'Particle Processing': [k for k in self.results.keys() if 'PartType' in k],
            'Training Splits': [k for k in self.results.keys() if k.startswith('splits')],
            'File Loading': [k for k in self.results.keys() if k.startswith('file_')]
        }
        
        for category, operations in categories.items():
            if operations:
                print(f"\n🔹 {category}:")
                
                successful_ops = [op for op in operations if self.results[op]['success']]
                
                if successful_ops:
                    durations = [self.results[op]['duration'] for op in successful_ops]
                    avg_duration = sum(durations) / len(durations)
                    min_duration = min(durations)
                    max_duration = max(durations)
                    
                    print(f"   Operations: {len(successful_ops)}/{len(operations)} successful")
                    print(f"   Avg time: {avg_duration:.2f}s")
                    print(f"   Range: {min_duration:.2f}s - {max_duration:.2f}s")
                    
                    # Show fastest and slowest
                    fastest = min(successful_ops, key=lambda x: self.results[x]['duration'])
                    slowest = max(successful_ops, key=lambda x: self.results[x]['duration'])
                    
                    print(f"   Fastest: {fastest} ({self.results[fastest]['duration']:.2f}s)")
                    print(f"   Slowest: {slowest} ({self.results[slowest]['duration']:.2f}s)")
        
        # Recommendations
        print("\n💡 Recommendations:")
        
        # Find optimal particle counts
        particle_ops = [k for k in self.results.keys() if 'PartType' in k and self.results[k]['success']]
        if particle_ops:
            fast_ops = [op for op in particle_ops if self.results[op]['duration'] < 2.0]
            if fast_ops:
                print("   🚀 Fast processing (< 2s):")
                for op in fast_ops[:3]:
                    print(f"     • {op}: {self.results[op]['duration']:.2f}s")
        
        # Total time estimate
        total_time = sum(r['duration'] for r in self.results.values() if r['success'])
        print(f"\n⏱️  Total benchmark time: {total_time:.2f}s")
        print(f"📊 Operations tested: {len(self.results)}")
        success_rate = len([r for r in self.results.values() if r['success']]) / len(self.results) * 100
        print(f"✅ Success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    benchmark = TNG50Benchmark()
    benchmark.run_full_benchmark() 