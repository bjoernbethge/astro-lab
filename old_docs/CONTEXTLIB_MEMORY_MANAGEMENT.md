# 🧠 AstroLab contextlib Memory Management

## Überblick

AstroLab verwendet umfassende `contextlib`-basierte Memory Management Patterns für optimale Ressourcenverwaltung in astronomischen Datenverarbeitungsanwendungen. Diese Dokumentation beschreibt die verfügbaren Context Manager und deren Verwendung.

## 📋 Inhaltsverzeichnis

- [Kernfeatures](#kernfeatures)
- [Verfügbare Context Manager](#verfügbare-context-manager)
- [Verwendungsbeispiele](#verwendungsbeispiele)
- [Best Practices](#best-practices)
- [Integration in eigene Projekte](#integration-in-eigene-projekte)

## 🚀 Kernfeatures

### Automatische Ressourcenverwaltung
- **PyTorch CUDA Cache Management**: Automatisches Leeren des CUDA-Caches
- **Garbage Collection**: Intelligente Speicherbereinigung
- **System Cache Cleanup**: Bereinigung von Python-internen Caches
- **Memory Tracking**: Detaillierte Speicherverbrauchsüberwachung

### Spezialisierte Context Manager
- **File Processing**: Optimierte Dateienverarbeitung mit Speicherlimits
- **Batch Processing**: Stapelverarbeitung mit adaptiver Speicherverwaltung  
- **Model Training**: ML-Training mit Memory-Optimierungen
- **Tensor Operations**: Speichereffiziente Tensor-Operationen

### Monitoring und Statistiken
- **Memory Statistics**: Detaillierte Speicherstatistiken
- **Continuous Monitoring**: Kontinuierliche Speicherüberwachung
- **Performance Tracking**: Leistungsmetriken und Optimierungshinweise

## 🛠️ Verfügbare Context Manager

### 1. Comprehensive Cleanup Context

```python
from astro_lab.utils.memory import comprehensive_cleanup_context

with comprehensive_cleanup_context(
    "Operation name",
    cleanup_pytorch=True,
    cleanup_matplotlib=True,
    cleanup_blender=True,
    force_gc=True
) as stats:
    # Ihre Operationen hier
    large_tensor = torch.randn(10000, 1000)
    result = some_heavy_computation(large_tensor)

# Automatische Bereinigung aller Ressourcen
print(f"Memory stats: {stats}")
```

**Features:**
- Automatische PyTorch CUDA Cache-Bereinigung
- Matplotlib Figure-Cleanup
- Blender Orphan Data-Bereinigung  
- Mehrfache Garbage Collection-Zyklen
- System Cache-Bereinigung

### 2. Memory Tracking Context

```python
from astro_lab.utils.memory import memory_tracking_context

with memory_tracking_context("Tensor operations") as stats:
    tensor1 = torch.randn(5000, 100)
    tensor2 = torch.randn(5000, 100)
    result = torch.matmul(tensor1, tensor2.T)

print(f"Objects created: {stats.object_diff}")
print(f"Memory used: {stats.memory_diff:.2f} MB")
print(f"Duration: {stats.duration:.2f} seconds")
```

**Tracked Metrics:**
- Anzahl erstellter/gelöschter Objekte
- Speicherverbrauch (System + CUDA)
- Peak Memory Usage
- Operationsdauer

### 3. File Processing Context

```python
from astro_lab.utils.memory import file_processing_context

with file_processing_context(
    file_path="large_catalog.parquet",
    memory_limit_mb=1000.0
) as params:
    # Optimale Chunk-Größe wird automatisch berechnet
    chunk_size = params['chunk_size']
    
    # Datei verarbeiten
    df = pl.read_parquet(params['file_path'])
    processed = preprocess_catalog(df)
```

**Features:**
- Automatische Chunk-Größen-Berechnung
- Speicherlimit-Überwachung
- Optimierte I/O-Parameter

### 4. Batch Processing Context

```python
from astro_lab.utils.memory import batch_processing_context

files = ["file1.parquet", "file2.parquet", "file3.parquet"]

with batch_processing_context(
    total_items=len(files),
    batch_size=2,
    memory_threshold_mb=500.0
) as batch_config:
    
    for file_path in files:
        with comprehensive_cleanup_context(f"Processing {file_path}"):
            # Jede Datei wird mit automatischer Bereinigung verarbeitet
            result = process_file(file_path)
```

**Features:**
- Adaptive Batch-Größen
- Memory Threshold Monitoring
- Automatische Bereinigung zwischen Batches

### 5. Model Training Context

```python
from astro_lab.utils.memory import model_training_context

with model_training_context(
    model_name="AstroNet",
    enable_mixed_precision=True,
    gradient_checkpointing=True
) as training_config:
    
    # Training mit Memory-Optimierungen
    model = create_model()
    trainer = AstroTrainer(model)
    trainer.fit(datamodule)
```

**Features:**
- Mixed Precision Training Setup
- Gradient Checkpointing
- Memory-efficient Attention
- CUDA Optimizations

### 6. PyTorch Memory Context

```python
from astro_lab.utils.memory import pytorch_memory_context

with pytorch_memory_context(
    "CUDA operations",
    clear_cache=True,
    reset_stats=True
) as initial_stats:
    
    # PyTorch-Operationen mit automatischer Cache-Verwaltung
    tensor = torch.randn(10000, 1000, device='cuda')
    result = torch.matmul(tensor, tensor.T)

# Cache wird automatisch geleert
```

### 7. Memory Monitor

```python
from astro_lab.utils.memory import MemoryMonitor

monitor = MemoryMonitor(interval=1.0, threshold_mb=1000.0)

with monitor.monitoring_context():
    # Kontinuierliche Speicherüberwachung
    for i in range(100):
        large_operation()

print(f"Peak memory: {monitor.max_memory:.2f} MB")
```

## 📊 Verwendungsbeispiele

### Beispiel 1: Astronomical Data Processing

```python
from astro_lab.utils.memory import comprehensive_cleanup_context
from astro_lab.data.manager import AstroDataManager
from astro_lab.tensors import Spatial3DTensor

# Große Katalogverarbeitung mit Memory Management
with comprehensive_cleanup_context("GAIA catalog processing"):
    # Data Manager mit automatischer Bereinigung
    manager = AstroDataManager()
    
    # Katalog laden
    catalog = manager.load_catalog("gaia_dr3_sample.parquet")
    
    # Spatial Tensor erstellen
    positions = catalog.select(['ra', 'dec', 'distance']).to_numpy()
    spatial_tensor = Spatial3DTensor(positions, unit="pc")
    
    # Memory-effiziente Operationen
    with spatial_tensor.memory_efficient_context("Clustering"):
        clusters = spatial_tensor.dbscan_clustering(eps=0.1, min_samples=5)
        
    print(f"Found {len(set(clusters))} clusters")

# Automatische Bereinigung aller Ressourcen
```

### Beispiel 2: Batch File Processing

```python
from astro_lab.utils.memory import batch_processing_context
from astro_lab.data.processing import EnhancedDataProcessor, SimpleProcessingConfig

# Konfiguration für Memory-optimierte Verarbeitung
config = SimpleProcessingConfig(
    memory_limit_mb=2000.0,
    enable_memory_optimization=True,
    cleanup_intermediate=True
)

processor = EnhancedDataProcessor(config)

# Batch-Verarbeitung mehrerer Surveys
survey_files = [
    "gaia_dr3.parquet",
    "sdss_dr17.parquet", 
    "nsa_catalog.parquet"
]

results = processor.process_batch(
    file_paths=survey_files,
    output_dir="processed_surveys"
)

print(f"Processed {results['total_objects']} objects")
print(f"Memory stats: {results['memory_stats']}")
```

### Beispiel 3: ML Training mit Memory Management

```python
from astro_lab.utils.memory import model_training_context
from astro_lab.training.trainer import AstroTrainer

with model_training_context(
    model_name="GalaxyClassifier",
    enable_mixed_precision=True
) as training_config:
    
    # Trainer mit automatischem Memory Management
    trainer = AstroTrainer(
        model="point_cloud_gnn",
        epochs=100,
        batch_size=32
    )
    
    # Training mit automatischer Bereinigung
    results = trainer.train()
    
    print(f"Best validation loss: {results['best_score']:.4f}")
    print(f"Training memory stats: {training_config['stats']}")
```

### Beispiel 4: Tensor Operations mit Batch Processing

```python
from astro_lab.tensors import FeatureTensor
from astro_lab.utils.memory import comprehensive_cleanup_context

# Große Feature-Matrix verarbeiten
n_objects = 100000
n_features = 500

with comprehensive_cleanup_context("Feature processing"):
    # Feature Tensor erstellen
    features = torch.randn(n_objects, n_features)
    feature_tensor = FeatureTensor(
        data=features,
        feature_names=[f"feature_{i}" for i in range(n_features)]
    )
    
    # Batch-Processing für Memory-Effizienz
    with feature_tensor.batch_processing_context(batch_size=1000) as batches:
        processed_batches = []
        
        for batch in batches:
            # Jeder Batch wird separat verarbeitet
            scaled_batch = (batch - batch.mean(dim=0)) / batch.std(dim=0)
            processed_batches.append(scaled_batch)
        
        # Batches zusammenführen
        processed_features = torch.cat(processed_batches, dim=0)
    
    print(f"Processed {processed_features.shape[0]} objects")

# Automatische Bereinigung
```

## 🎯 Best Practices

### 1. Context Manager Hierarchie

```python
# Äußerer Context für gesamte Operation
with comprehensive_cleanup_context("Data pipeline"):
    
    # Spezialisierte Contexts für Teiloperationen
    with file_processing_context(file_path, memory_limit_mb=1000):
        data = load_data(file_path)
    
    with pytorch_memory_context("Model inference"):
        predictions = model(data)
    
    with memory_tracking_context("Results processing") as stats:
        results = postprocess_results(predictions)
        
print(f"Pipeline completed: {stats}")
```

### 2. Memory Monitoring für kritische Operationen

```python
from astro_lab.utils.memory import MemoryMonitor

monitor = MemoryMonitor(threshold_mb=2000.0)

with monitor.monitoring_context():
    with comprehensive_cleanup_context("Critical operation"):
        # Memory-intensive Operation
        result = heavy_computation()

if monitor.max_memory > 1500:
    print(f"⚠️ High memory usage detected: {monitor.max_memory:.2f} MB")
```

### 3. Adaptive Batch Sizes

```python
def process_with_adaptive_batching(data, initial_batch_size=1000):
    with memory_tracking_context("Adaptive batching") as stats:
        batch_size = initial_batch_size
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            
            with pytorch_memory_context(f"Batch {i//batch_size}"):
                process_batch(batch)
            
            # Batch-Größe anpassen basierend auf Memory Usage
            if stats.memory_diff > 500:  # > 500 MB
                batch_size = max(batch_size // 2, 100)
            elif stats.memory_diff < 100:  # < 100 MB  
                batch_size = min(batch_size * 2, 5000)
```

### 4. Error Handling mit Context Managern

```python
from astro_lab.utils.memory import comprehensive_cleanup_context

def robust_processing(data_files):
    successful = []
    failed = []
    
    for file_path in data_files:
        try:
            with comprehensive_cleanup_context(f"Processing {file_path}"):
                result = process_file(file_path)
                successful.append((file_path, result))
                
        except Exception as e:
            failed.append((file_path, str(e)))
            # Context Manager sorgt trotzdem für Cleanup
    
    return successful, failed
```

## 🔧 Integration in eigene Projekte

### 1. Custom Context Manager erstellen

```python
from astro_lab.utils.memory import comprehensive_cleanup_context
from contextlib import contextmanager

@contextmanager
def custom_astronomy_context(operation_name, survey_name=None):
    """Custom Context Manager für astronomische Operationen."""
    
    with comprehensive_cleanup_context(f"{operation_name} - {survey_name}"):
        # Setup
        print(f"🔭 Starting {operation_name} for {survey_name}")
        
        # Custom setup logic
        setup_astronomy_environment()
        
        try:
            yield
        finally:
            # Custom cleanup logic
            cleanup_astronomy_resources()
            print(f"✅ Completed {operation_name}")

# Verwendung
with custom_astronomy_context("Galaxy classification", "SDSS"):
    # Ihre astronomischen Operationen
    pass
```

### 2. Memory-optimierte Klassen

```python
from astro_lab.utils.memory import comprehensive_cleanup_context

class MemoryOptimizedProcessor:
    """Beispiel für Memory-optimierte Klasse."""
    
    def __init__(self):
        self.memory_limit_mb = 1000.0
    
    def process_survey(self, survey_data):
        with comprehensive_cleanup_context(f"Survey processing"):
            return self._internal_processing(survey_data)
    
    def _internal_processing(self, data):
        # Implementierung mit automatischer Bereinigung
        pass
```

### 3. CLI Integration

```python
# In Ihrem CLI-Code
from astro_lab.utils.memory import comprehensive_cleanup_context

@click.command()
@click.argument('input_file')
def process_command(input_file):
    """CLI Command mit Memory Management."""
    
    with comprehensive_cleanup_context("CLI processing"):
        try:
            result = process_file(input_file)
            click.echo(f"✅ Processing completed: {result}")
            
        except Exception as e:
            click.echo(f"❌ Processing failed: {e}")
            sys.exit(1)
    
    # Automatische Bereinigung auch bei Fehlern
```

## 📈 Performance Tipps

### 1. Memory Limits setzen

```python
# Für große Dateien
with file_processing_context(file_path, memory_limit_mb=2000.0):
    # Automatische Chunk-Größen-Optimierung
    pass

# Für Batch Processing  
with batch_processing_context(total_items=1000, memory_threshold_mb=1500.0):
    # Adaptive Batch-Größen
    pass
```

### 2. Lazy Loading verwenden

```python
with comprehensive_cleanup_context("Lazy loading"):
    # Lazy imports für bessere Memory-Effizienz
    from astro_lab.heavy_module import HeavyClass
    
    # Nur bei Bedarf laden
    if condition:
        heavy_object = HeavyClass()
        result = heavy_object.process()
```

### 3. Context Manager verschachteln

```python
# Optimale Verschachtelung für maximale Effizienz
with comprehensive_cleanup_context("Main operation"):
    with memory_tracking_context("Data loading") as load_stats:
        data = load_large_dataset()
    
    with pytorch_memory_context("GPU processing"):
        with memory_tracking_context("Model inference") as inference_stats:
            results = model(data)
    
    print(f"Loading: {load_stats.memory_diff:.2f} MB")
    print(f"Inference: {inference_stats.memory_diff:.2f} MB")
```

## 🚨 Troubleshooting

### Memory Leaks identifizieren

```python
from astro_lab.utils.memory import memory_tracking_context

with memory_tracking_context("Leak detection") as stats:
    # Verdächtige Operation
    suspicious_operation()

if stats.object_diff > 1000:
    print(f"⚠️ Possible memory leak: {stats.object_diff} objects not freed")
```

### CUDA Out of Memory

```python
from astro_lab.utils.memory import pytorch_memory_context

try:
    with pytorch_memory_context("CUDA operation", clear_cache=True):
        result = cuda_intensive_operation()
        
except RuntimeError as e:
    if "out of memory" in str(e):
        print("💾 CUDA out of memory - reducing batch size")
        # Fallback mit kleinerer Batch-Größe
        result = process_with_smaller_batches()
```

### Performance Monitoring

```python
from astro_lab.utils.memory import MemoryMonitor

monitor = MemoryMonitor(interval=0.5, threshold_mb=1000.0)

with monitor.monitoring_context():
    # Operation überwachen
    long_running_operation()

# Performance-Analyse
if monitor.max_memory > 2000:
    print("🐌 High memory usage - consider optimization")
```

## 📚 Weitere Ressourcen

- [Memory Management Best Practices](MEMORY_BEST_PRACTICES.md)
- [PyTorch Memory Optimization](PYTORCH_MEMORY.md)  
- [Astronomical Data Processing Guide](DATA_PROCESSING.md)
- [Performance Tuning](PERFORMANCE_TUNING.md)

## 🤝 Beitrag leisten

Haben Sie Verbesserungsvorschläge für die Memory Management Features? 

1. Fork das Repository
2. Erstellen Sie einen Feature Branch
3. Implementieren Sie Ihre Verbesserungen
4. Erstellen Sie einen Pull Request

Wir freuen uns über Beiträge zur Verbesserung der Memory-Effizienz!

---

*Diese Dokumentation wird kontinuierlich aktualisiert. Letzte Aktualisierung: $(date)* 