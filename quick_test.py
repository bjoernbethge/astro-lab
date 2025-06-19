#!/usr/bin/env python3
import time
from pathlib import Path
from astro_lab.data import import_tng50, load_catalog, create_training_splits

print('🚀 Quick TNG50 Performance Test')
print('=' * 40)

snap_file = Path('data/raw/TNG50-4/output/snapdir_099/snap_099.0.hdf5')

# Test 1: Import und Load für PartType4 (Stars)
start = time.time()
parquet_file = import_tng50(snap_file, 'PartType4')
df = load_catalog(parquet_file)
import_time = time.time() - start
print(f'📊 PartType4 Import+Load: {import_time:.2f}s ({len(df):,} particles)')

# Test 2: Sample verschiedene Größen
sizes = [1000, 5000, 10000, 20000]
for size in sizes:
    if len(df) >= size:
        start = time.time()
        df_sample = df.sample(size, seed=42)
        sample_time = time.time() - start
        print(f'   Sample {size:,}: {sample_time:.3f}s')

# Test 3: Training Splits für 10k Dataset
start = time.time()
df_10k = df.sample(10000, seed=42)
train, val, test = create_training_splits(df_10k, test_size=0.2, val_size=0.1)
split_time = time.time() - start
print(f'🔄 Training splits (10k): {split_time:.3f}s')
print(f'   Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}')

# Test 4: Black Holes (PartType5)
start = time.time()
parquet_file_bh = import_tng50(snap_file, 'PartType5')
df_bh = load_catalog(parquet_file_bh)
bh_time = time.time() - start
print(f'🕳️  PartType5 Import+Load: {bh_time:.3f}s ({len(df_bh):,} black holes)')

print(f'\n⏱️  Total test time: {import_time + split_time + bh_time:.2f}s') 