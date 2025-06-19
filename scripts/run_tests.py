#!/usr/bin/env python3
"""
Test runner for astro-lab with better memory management.

Suppresses known fake-bpy-module memory leak warnings.
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Run tests with proper environment setup."""
    # Set environment variables to suppress known fake-bpy-module memory leaks
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '0'  # Deterministic hash seed
    env['MALLOC_CHECK_'] = '0'  # Suppress malloc checks that cause false positives
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run pytest with our custom environment
    cmd = [sys.executable, "-m", "pytest"] + sys.argv[1:]
    
    print("🧪 Running astro-lab tests with optimized memory management...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run with filtered output
    try:
        # Create a temporary file for stderr
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as stderr_file:
            stderr_file_path = stderr_file.name
        
        # Run pytest with stderr redirected
        with open(stderr_file_path, 'w') as stderr_f:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=stderr_f,
                text=True
            )
        
        # Print stdout normally
        print(result.stdout)
        
        # Read and filter stderr
        try:
            with open(stderr_file_path, 'r') as stderr_f:
                stderr_content = stderr_f.read()
            
            # Filter out fake-bpy-module memory leak messages
            stderr_lines = stderr_content.split('\n')
            filtered_stderr = []
            
            skip_next = False
            for line in stderr_lines:
                # Skip the memory leak error and the line after it
                if "Error: Not freed memory blocks" in line:
                    skip_next = True
                    continue
                if skip_next and "total unfreed memory" in line:
                    skip_next = False
                    continue
                if skip_next:
                    skip_next = False
                
                # Keep other errors
                if line.strip():
                    filtered_stderr.append(line)
            
            # Print filtered stderr only if it contains actual errors
            if filtered_stderr:
                print('\n'.join(filtered_stderr), file=sys.stderr)
        
        finally:
            # Clean up temp file
            try:
                os.unlink(stderr_file_path)
            except:
                pass
        
        # Return original exit code
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 