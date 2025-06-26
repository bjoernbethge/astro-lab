# Memory Management in AstroLab

This document describes the robust and DRY memory management strategies implemented in AstroLab to prevent memory leaks and optimize performance.

## Overview

AstroLab implements a centralized, robust memory management system based on best practices for Python applications, particularly for machine learning and data processing workloads. The implementation follows DRY principles and provides comprehensive protection against memory leaks.

## Architecture

### Centralized Memory Manager

The core of AstroLab's memory management is the `MemoryManager` class in `astro_lab.memory`:

```python
from astro_lab.memory import MemoryManager

# Global memory manager instance
_memory_manager = MemoryManager()
```

### Key Features

1. **Centralized Management**: Single point of control for all memory operations
2. **Automatic Cleanup**: Context managers ensure proper cleanup
3. **Memory Tracking**: Real-time memory usage monitoring
4. **Weak References**: Prevents circular references
5. **CUDA Integration**: Automatic GPU memory management

## Memory Management Strategies

### 1. Context Manager Pattern

The primary interface for memory management:

```python
from astro_lab.memory import memory_management

with memory_management():
    # Your operations here
    result = process_large_dataset()
    # Automatic cleanup happens here
```

### 2. Memory Statistics

Real-time memory monitoring:

```python
from astro_lab.memory import get_memory_stats

stats = get_memory_stats()
print(f"Current: {stats['current_mb']:.1f}MB")
print(f"CUDA: {stats['cuda_memory_allocated']:.1f}MB")
```

### 3. Memory Diagnosis

Automatic detection of potential issues:

```python
from astro_lab.memory import diagnose_memory_leaks

diagnosis = diagnose_memory_leaks()
print("Recommendations:", diagnosis["recommendations"])
```

### 4. Comprehensive Cleanup

Force cleanup when needed:

```python
from astro_lab.memory import force_comprehensive_cleanup

result = force_comprehensive_cleanup()
print(f"Cleanup completed: {result['success']}")
```

## TensorDict Memory Management

The `AstroTensorDict` class includes built-in memory management:

### Automatic Cleanup

```python
from astro_lab.tensors import SurveyTensorDict

# TensorDict automatically registers for cleanup
tensor_dict = SurveyTensorDict(...)

# Explicit cleanup
tensor_dict.cleanup()

# Memory optimization
tensor_dict.optimize_memory()
```

### Memory Information

```python
# Get memory usage information
info = tensor_dict.memory_info()
print(f"Total memory: {info['total_mb']:.1f}MB")
print(f"Number of tensors: {info['n_tensors']}")
```

## UI Memory Management

### Memory Management Panel

The UI includes a dedicated memory management panel:

```python
from astro_lab.ui.components import ui_memory_management

# Get memory management UI
memory_ui = ui_memory_management()
```

### System Status

Enhanced system status with memory information:

```python
from astro_lab.ui.components import ui_system_status

# Get system status with memory controls
status_ui = ui_system_status()
```

## Implementation Details

### Memory Manager Class

```python
class MemoryManager:
    """Centralized memory management for AstroLab."""
    
    @contextmanager
    def context(self):
        """Context manager for memory-efficient operations."""
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        
    def diagnose(self) -> Dict[str, Any]:
        """Diagnose potential memory issues."""
        
    def force_cleanup(self) -> Dict[str, Any]:
        """Force comprehensive memory cleanup."""
```

### TensorDict Integration

```python
class AstroTensorDict(TensorDict):
    """Basis-class with memory management."""
    
    def __init__(self, data: Dict[str, Any], **kwargs):
        # Register for memory management
        register_for_cleanup(self)
    
    def cleanup(self):
        """Explicit cleanup method."""
        
    def optimize_memory(self):
        """Optimize memory usage."""
```

## Best Practices

### 1. Use Context Managers

Always use memory management context managers for large operations:

```python
with memory_management():
    # Large data processing
    result = process_astronomical_data()
```

### 2. Monitor Memory Usage

Regularly check memory statistics:

```python
stats = get_memory_stats()
if stats["current_mb"] > 1000:  # > 1GB
    print("High memory usage detected")
```

### 3. Clean Up Large Objects

Explicitly clean up large objects when no longer needed:

```python
# Large dataset
dataset = load_large_dataset()

# Process data
results = process_dataset(dataset)

# Clean up
dataset.cleanup()
del dataset
```

### 4. Use TensorDict Memory Methods

Leverage built-in TensorDict memory management:

```python
# Optimize memory
tensor_dict.optimize_memory()

# Clear temporary tensors
tensor_dict.clear_temp_tensors()

# Get memory info
info = tensor_dict.memory_info()
```

### 5. Monitor CUDA Memory

Regularly monitor CUDA memory usage:

```python
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    print(f"CUDA Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
```

## Memory Leak Detection

### Automatic Detection

AstroLab automatically detects potential memory leaks:

1. **High Object Counts**: Objects with count > 1000 are flagged
2. **High Memory Usage**: Usage > 500MB triggers warnings
3. **CUDA Memory**: Usage > 1GB triggers cleanup recommendations

### Manual Diagnosis

```python
diagnosis = diagnose_memory_leaks()
print("Top memory consumers:", diagnosis["top_consumers"])
print("Recommendations:", diagnosis["recommendations"])
```

## Performance Monitoring

### Memory Statistics

```python
stats = get_memory_stats()
print(f"Current: {stats['current_mb']:.1f}MB")
print(f"Peak: {stats['peak_mb']:.1f}MB")
print(f"CUDA: {stats['cuda_memory_allocated']:.1f}MB")
```

### Widget Status

```python
status = get_widget_status()
print(f"Widget available: {status['available']}")
print(f"CUDA available: {status['torch_available']}")
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Use `force_comprehensive_cleanup()`
2. **CUDA Out of Memory**: Clear CUDA cache and reduce batch sizes
3. **Slow Performance**: Check for memory leaks with `diagnose_memory_leaks()`

### Debug Mode

Enable debug logging for detailed memory information:

```python
import logging
logging.getLogger('astro_lab.memory').setLevel(logging.DEBUG)
```

## API Reference

### Memory Manager (`astro_lab.memory`)

- `memory_management()`: Context manager for memory operations
- `get_memory_stats()`: Get current memory statistics
- `diagnose_memory_leaks()`: Diagnose potential memory issues
- `force_comprehensive_cleanup()`: Force comprehensive cleanup
- `register_for_cleanup()`: Register object for memory tracking

### TensorDict (`astro_lab.tensors`)

- `cleanup()`: Explicit cleanup method
- `optimize_memory()`: Optimize memory usage
- `clear_temp_tensors()`: Clear temporary tensors
- `memory_info()`: Get memory information

### UI Components (`astro_lab.ui.components`)

- `ui_memory_management()`: Memory management UI panel
- `ui_system_status()`: System status with memory controls

## File Structure

```
src/astro_lab/
├── memory.py              # Centralized memory management
├── ui/
│   └── components.py      # UI components (imports memory)
└── tensors/
    └── tensordict_astro.py # TensorDict with memory integration
```

## References

- [Python Memory Management](https://www.geeksforgeeks.org/python/diagnosing-and-fixing-memory-leaks-in-python/)
- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [TensorDict Documentation](https://pytorch.org/tensordict/)

## Conclusion

AstroLab's robust memory management system provides comprehensive protection against memory leaks while maintaining high performance. The centralized approach ensures consistency and the DRY implementation reduces code duplication. By following the best practices outlined in this document, users can ensure optimal memory usage in their astronomical data processing workflows. 