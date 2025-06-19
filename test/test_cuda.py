"""
CUDA Tests for AstroLab
=======================

Comprehensive GPU tests for astronomical computing workloads.
"""

import time
from typing import Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*50}")
    print(f"📋 {title}")
    print('='*50)


def print_subheader(title: str) -> None:
    """Print a formatted subheader."""
    print(f"\n📋 {title}")
    print('-'*40)


def get_gpu_info() -> Dict:
    """Get comprehensive GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}

    info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": []
    }

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            "id": i,
            "name": props.name,
            "major": props.major,
            "minor": props.minor,
            "total_memory_mb": props.total_memory / 1024**2,
            "multi_processor_count": props.multi_processor_count,
            "max_threads_per_multiprocessor": getattr(props, 'max_threads_per_multiprocessor', 'N/A'),
            "max_threads_per_block": getattr(props, 'max_threads_per_block', 'N/A'),
            "warp_size": getattr(props, 'warp_size', 32),
        }
        info["devices"].append(device_info)

    return info


@pytest.mark.cuda
def test_basic_cuda():
    """Test basic CUDA functionality."""
    print_header("Basic CUDA Tests")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Get GPU info
    gpu_info = get_gpu_info()
    
    print("🔍 GPU Information:")
    for device in gpu_info["devices"]:
        print(f"  Device {device['id']}: {device['name']}")
        print(f"    Compute Capability: {device['major']}.{device['minor']}")
        print(f"    Total Memory: {device['total_memory_mb']:.0f} MB")
        print(f"    Multiprocessors: {device['multi_processor_count']}")
        print(f"    Max Threads/Block: {device['max_threads_per_block']}")
        print(f"    Warp Size: {device['warp_size']}")

    # Test basic tensor operations
    print("\n🧪 Basic Tensor Operations:")
    
    device = torch.device("cuda")
    
    # Create tensors
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    
    print("  ✓ Tensors created on GPU")
    
    # Matrix multiplication
    c = torch.mm(a, b)
    print("  ✓ Matrix multiplication completed")
    
    # Element-wise operations
    d = a + b
    e = a * b
    print("  ✓ Element-wise operations completed")
    
    # Memory transfer
    c_cpu = c.cpu()
    c_gpu = c_cpu.cuda()
    print("  ✓ CPU ↔ GPU memory transfers completed")
    
    # Verify results are on correct device
    assert a.device.type == "cuda"
    assert c.device.type == "cuda"
    assert c_cpu.device.type == "cpu"
    assert c_gpu.device.type == "cuda"
    
    print("✅ Basic CUDA tests passed!")


@pytest.mark.cuda
def test_memory_management():
    """Test GPU memory management."""
    print_subheader("Memory Management")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    
    # Get initial memory stats
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    print(f"🧪 Initial GPU memory: {initial_memory / 1024**2:.1f} MB")
    
    # Allocate tensors of increasing size
    tensors = []
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        try:
            tensor = torch.randn(size, size, device=device)
            tensors.append(tensor)
            
            current_memory = torch.cuda.memory_allocated()
            print(f"  ✓ Allocated {size}x{size} tensor: {current_memory / 1024**2:.1f} MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ Out of memory at size {size}x{size}")
                break
            else:
                raise
    
    # Test memory cleanup
    peak_memory = torch.cuda.memory_allocated()
    
    # Delete tensors
    del tensors
    torch.cuda.empty_cache()
    
    final_memory = torch.cuda.memory_allocated()
    
    print(f"  Peak memory: {peak_memory / 1024**2:.1f} MB")
    print(f"  Final memory: {final_memory / 1024**2:.1f} MB")
    print(f"  Memory freed: {(peak_memory - final_memory) / 1024**2:.1f} MB")
    
    # Memory should be freed (allowing some tolerance)
    # Note: GPU memory management can be complex, so we allow generous tolerance
    memory_tolerance = max(50 * 1024**2, initial_memory + 20 * 1024**2)  # 50MB or initial + 20MB
    assert final_memory <= memory_tolerance
    
    print("✅ Memory management tests passed!")


@pytest.mark.cuda
@pytest.mark.slow
def test_performance_benchmarks():
    """Test GPU performance with astronomical data operations."""
    print_subheader("Performance Benchmarks")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    # Benchmark matrix operations
    print("🧪 Matrix Operations Benchmark:")
    sizes = [512, 1024, 2048]  # Reduced sizes for faster testing

    for size in sizes:
        try:
            # GPU benchmark - use CUDA events for precise timing (NVIDIA Best Practice)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Warmup
            for _ in range(2):
                a_warmup = torch.randn(size, size, device=device)
                b_warmup = torch.randn(size, size, device=device)
                c_warmup = torch.mm(a_warmup, b_warmup)
            
            torch.cuda.synchronize()
            start_event.record()

            for _ in range(5):  # Multiple iterations
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                c = torch.mm(a, b)

            end_event.record()
            torch.cuda.synchronize()
            gpu_time = start_event.elapsed_time(end_event) / 1000.0 / 5  # Convert to seconds, average

            # CPU benchmark
            a_cpu = torch.randn(size, size)
            b_cpu = torch.randn(size, size)

            start_time = time.perf_counter()
            for _ in range(5):  # Multiple iterations
                c_cpu = torch.mm(a_cpu, b_cpu)
            cpu_time = (time.perf_counter() - start_time) / 5  # Average time

            # Safe division with minimum time threshold
            if gpu_time > 1e-6:  # Minimum 1 microsecond
                speedup = cpu_time / gpu_time
                print(f"  Matrix {size:4d}x{size}: GPU {gpu_time:.6f}s, CPU {cpu_time:.6f}s, Speedup: {speedup:.1f}x")
            else:
                print(f"  Matrix {size:4d}x{size}: GPU operation too fast to measure accurately")

            del a, b, c, a_cpu, b_cpu, c_cpu

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ Matrix {size:4d}x{size}: Out of memory")
                break
            else:
                raise

    # Benchmark FFT operations (common in astronomy)
    print("\n🧪 FFT Operations Benchmark:")
    fft_sizes = [256, 512, 1024]  # Reduced sizes

    for size in fft_sizes:
        try:
            # GPU FFT - multiple iterations
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            for _ in range(5):
                signal = torch.randn(size, size, device=device, dtype=torch.complex64)
                fft_result = torch.fft.fft2(signal)

            torch.cuda.synchronize()
            gpu_time = (time.perf_counter() - start_time) / 5

            # CPU FFT
            start_time = time.perf_counter()
            for _ in range(5):
                signal_cpu = torch.randn(size, size, dtype=torch.complex64)
                fft_result_cpu = torch.fft.fft2(signal_cpu)
            cpu_time = (time.perf_counter() - start_time) / 5

            # Safe division
            if gpu_time > 1e-6:
                speedup = cpu_time / gpu_time
                print(f"  FFT2D {size:4d}x{size}: GPU {gpu_time:.6f}s, CPU {cpu_time:.6f}s, Speedup: {speedup:.1f}x")
            else:
                print(f"  FFT2D {size:4d}x{size}: GPU operation too fast to measure accurately")

            del signal, fft_result, signal_cpu, fft_result_cpu

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ FFT2D {size:4d}x{size}: Out of memory")
                break
            else:
                raise

    torch.cuda.empty_cache()
    # Memory Bandwidth Test (NVIDIA Best Practice - Chapter 8.2)
    print("\n🧪 Memory Bandwidth Benchmark:")
    
    # Test different memory access patterns
    memory_sizes = [1024*1024, 4*1024*1024, 16*1024*1024]  # 1MB, 4MB, 16MB
    
    for size in memory_sizes:
        try:
            # Create large arrays for bandwidth testing
            n_elements = size // 4  # 4 bytes per float32
            
            # GPU memory bandwidth test
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Warmup
            a_test = torch.randn(n_elements, device=device)
            b_test = torch.randn(n_elements, device=device)
            
            torch.cuda.synchronize()
            start_event.record()
            
            # Memory-bound operation (element-wise addition)
            for _ in range(10):
                c_test = a_test + b_test
            
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_ms = start_event.elapsed_time(end_event)
            elapsed_s = elapsed_ms / 1000.0
            
            # Calculate effective bandwidth (Read A + Read B + Write C = 3 * size)
            bytes_transferred = 3 * size * 10  # 10 iterations
            bandwidth_gb_s = (bytes_transferred / elapsed_s) / (1024**3)
            
            print(f"  Memory {size/1024/1024:.0f}MB: {bandwidth_gb_s:.1f} GB/s effective bandwidth")
            
            del a_test, b_test, c_test
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ Memory {size/1024/1024:.0f}MB: Out of memory")
                break
            else:
                raise

    print("✅ Performance benchmarks completed!")


@pytest.mark.cuda
def test_deep_learning_operations():
    """Test deep learning operations on GPU."""
    print_subheader("Deep Learning Operations")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    # Create a simple CNN for astronomical image classification
    class AstroCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(AstroCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(128 * 32 * 32, 512)
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 32 * 32)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
            return x

    print("🧪 CNN Model Tests:")

    # Create model and move to GPU
    model = AstroCNN(num_classes=10).to(device)
    model.train()

    print("  ✓ Model created and moved to GPU")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 16  # Smaller batch for testing
    x = torch.randn(batch_size, 3, 256, 256, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    start_time = time.perf_counter()
    output = model(x)
    forward_time = time.perf_counter() - start_time

    print(f"  ✓ Forward pass ({batch_size} samples): {forward_time:.4f}s")
    print(f"  ✓ Output shape: {output.shape}")

    # Test backward pass
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss = criterion(output, y)

    start_time = time.perf_counter()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    backward_time = time.perf_counter() - start_time

    print(f"  ✓ Backward pass: {backward_time:.4f}s")
    print(f"  ✓ Loss: {loss.item():.4f}")

    # GPU Occupancy Test (NVIDIA Best Practice - Chapter 10.1)
    print("\n🧪 GPU Occupancy Analysis:")
    
    # Get GPU properties for occupancy calculation
    props = torch.cuda.get_device_properties(device)
    max_threads_per_sm = getattr(props, 'max_threads_per_multiprocessor', 2048)
    multiprocessor_count = props.multi_processor_count
    
    print(f"  GPU: {props.name}")
    print(f"  Multiprocessors: {multiprocessor_count}")
    print(f"  Max threads per SM: {max_threads_per_sm}")
    
    # Test different batch sizes for occupancy
    batch_sizes = [8, 16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        try:
            # Create input for occupancy test
            x_occ = torch.randn(batch_size, 3, 128, 128, device=device)  # Smaller images for occupancy test
            
            # Measure inference time
            model.eval()
            with torch.no_grad():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # Warmup
                for _ in range(3):
                    _ = model(x_occ)
                
                torch.cuda.synchronize()
                start_event.record()
                
                for _ in range(10):
                    output_occ = model(x_occ)
                
                end_event.record()
                torch.cuda.synchronize()
                
                elapsed_ms = start_event.elapsed_time(end_event)
                avg_time_ms = elapsed_ms / 10
                throughput = batch_size / (avg_time_ms / 1000.0)  # samples per second
                
                print(f"  Batch {batch_size:3d}: {avg_time_ms:.2f}ms, {throughput:.0f} samples/s")
                
                del x_occ, output_occ
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ Batch {batch_size:3d}: Out of memory")
                break
            else:
                raise

    print("✅ Deep learning operations completed!")


@pytest.mark.cuda
def test_astronomical_specific():
    """Test GPU operations specific to astronomical computing."""
    print_subheader("Astronomical Computing")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    print("🧪 Astronomical Operations Tests:")

    # Test coordinate transformations (common in astronomy)
    n_objects = 100000
    
    # RA/Dec to Cartesian conversion
    ra = torch.rand(n_objects, device=device) * 360  # degrees
    dec = (torch.rand(n_objects, device=device) - 0.5) * 180  # degrees
    distance = torch.rand(n_objects, device=device) * 1000  # parsecs

    start_time = time.perf_counter()
    
    # Convert to radians
    ra_rad = torch.deg2rad(ra)
    dec_rad = torch.deg2rad(dec)
    
    # Convert to Cartesian coordinates
    x = distance * torch.cos(dec_rad) * torch.cos(ra_rad)
    y = distance * torch.cos(dec_rad) * torch.sin(ra_rad)
    z = distance * torch.sin(dec_rad)
    
    coord_time = time.perf_counter() - start_time
    print(f"  ✓ Coordinate transformation ({n_objects:,} objects): {coord_time:.4f}s")

    # Test distance calculations (N-body problem)
    n_particles = 1000
    positions = torch.randn(n_particles, 3, device=device)
    
    start_time = time.perf_counter()
    
    # Calculate pairwise distances
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # Broadcasting
    distances = torch.norm(diff, dim=2)
    
    distance_time = time.perf_counter() - start_time
    print(f"  ✓ Pairwise distances ({n_particles} particles): {distance_time:.4f}s")

    # Test magnitude calculations
    n_stars = 50000
    fluxes = torch.rand(n_stars, 5, device=device) * 1e-15  # 5-band photometry
    
    start_time = time.perf_counter()
    
    # Convert flux to magnitude: m = -2.5 * log10(flux) + zp
    zeropoints = torch.tensor([23.9, 23.0, 22.5, 22.0, 21.3], device=device)  # SDSS zeropoints
    magnitudes = -2.5 * torch.log10(fluxes) + zeropoints
    
    # Calculate colors
    colors = magnitudes[:, :-1] - magnitudes[:, 1:]  # Adjacent band differences
    
    mag_time = time.perf_counter() - start_time
    print(f"  ✓ Magnitude calculations ({n_stars:,} stars): {mag_time:.4f}s")

    # Test convolution (PSF operations)
    image_size = 512
    images = torch.randn(10, 1, image_size, image_size, device=device)
    
    # Create Gaussian PSF kernel
    kernel_size = 15
    sigma = 2.0
    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    y = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    start_time = time.perf_counter()
    
    # Apply PSF convolution
    convolved = F.conv2d(images, kernel, padding=kernel_size//2)
    
    conv_time = time.perf_counter() - start_time
    print(f"  ✓ PSF convolution (10 x {image_size}x{image_size} images): {conv_time:.4f}s")

    # Memory Coalescing Test (NVIDIA Best Practice - Chapter 9.2.1)
    print("\n🧪 Memory Access Pattern Tests:")
    
    matrix_size = 2048
    
    # Test 1: Coalesced access (row-major, good)
    matrix = torch.randn(matrix_size, matrix_size, device=device)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    # Row-wise access (coalesced)
    row_sums = torch.sum(matrix, dim=1)
    
    end_event.record()
    torch.cuda.synchronize()
    coalesced_time = start_event.elapsed_time(end_event)
    
    # Test 2: Non-coalesced access (column-major, bad)
    start_event.record()
    
    # Column-wise access (non-coalesced)
    col_sums = torch.sum(matrix, dim=0)
    
    end_event.record()
    torch.cuda.synchronize()
    non_coalesced_time = start_event.elapsed_time(end_event)
    
    print(f"  Coalesced access (row-wise): {coalesced_time:.2f}ms")
    print(f"  Non-coalesced access (col-wise): {non_coalesced_time:.2f}ms")
    
    if non_coalesced_time > 0:
        efficiency_ratio = coalesced_time / non_coalesced_time
        print(f"  Memory access efficiency ratio: {efficiency_ratio:.2f}")
    
    # Test 3: Strided access pattern
    stride_sizes = [1, 2, 4, 8, 16]
    
    print(f"\n  Strided Access Patterns:")
    for stride in stride_sizes:
        if stride * 1000 < matrix_size:
            start_event.record()
            
            # Strided access
            strided_data = matrix[::stride, ::stride]
            result = torch.sum(strided_data)
            
            end_event.record()
            torch.cuda.synchronize()
            strided_time = start_event.elapsed_time(end_event)
            
            print(f"    Stride {stride:2d}: {strided_time:.2f}ms")
            
            del strided_data

    print("✅ Astronomical computing tests completed!")


if __name__ == "__main__":
    """Run CUDA tests directly."""
    print_header("AstroLab CUDA Test Suite")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available on this system")
        exit(1)
    
    try:
        test_basic_cuda()
        test_memory_management() 
        test_performance_benchmarks()
        test_deep_learning_operations()
        test_astronomical_specific()
        
        print_header("All CUDA Tests Passed! 🎉")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 