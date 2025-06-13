#!/usr/bin/env python3
"""
Example script demonstrating the new codec abstraction system.

This shows how to use different raw data codecs for non-image data:
1. pickle_raw (legacy behavior) - each data point is pickled individually
2. pyarrow_batch - batches data points for better seeking performance
"""

import numpy as np
import tempfile
import os
from pathlib import Path

# Add the project directory to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from robodm import Trajectory, FeatureType
from robodm.backend.codec_config import CodecConfig

def demo_pickle_codec():
    """Demonstrate the pickle-based raw codec (legacy behavior)"""
    print("=== Pickle Raw Codec Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "pickle_demo.vla")
        
        # Create trajectory with pickle-based raw codec
        traj = Trajectory(path, mode="w", video_codec="rawvideo_pickle")
        
        # Add some test data
        for i in range(10):
            # Non-image data - will use raw codec
            vector_data = np.random.rand(5).astype(np.float32)
            joint_positions = np.array([i, i+1, i+2], dtype=np.float32)
            
            traj.add("sensor/vector", vector_data, timestamp=i*100)
            traj.add("robot/joints", joint_positions, timestamp=i*100)
        
        traj.close()
        
        # Read back and verify
        traj_read = Trajectory(path, mode="r")
        data = traj_read.load()
        traj_read.close()
        
        print(f"Loaded {len(data)} features:")
        for key, values in data.items():
            print(f"  {key}: shape={values.shape}, dtype={values.dtype}")
        
        file_size = os.path.getsize(path)
        print(f"File size: {file_size} bytes")
        
        return file_size


def demo_pyarrow_codec():
    """Demonstrate the PyArrow-based raw codec with batching"""
    print("\n=== PyArrow Batch Codec Demo ===")
    
    try:
        import pyarrow  # Check if PyArrow is available
    except ImportError:
        print("PyArrow not available - skipping demo")
        return None
    
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "pyarrow_demo.vla")
        
        # Create trajectory with PyArrow-based raw codec
        traj = Trajectory(path, mode="w", video_codec="rawvideo_pyarrow")
        
        # Add the same test data
        for i in range(10):
            # Non-image data - will use raw codec
            vector_data = np.random.rand(5).astype(np.float32)
            joint_positions = np.array([i, i+1, i+2], dtype=np.float32)
            
            traj.add("sensor/vector", vector_data, timestamp=i*100)
            traj.add("robot/joints", joint_positions, timestamp=i*100)
        
        traj.close()
        
        # Read back and verify
        traj_read = Trajectory(path, mode="r")
        data = traj_read.load()
        traj_read.close()
        
        print(f"Loaded {len(data)} features:")
        for key, values in data.items():
            print(f"  {key}: shape={values.shape}, dtype={values.dtype}")
        
        file_size = os.path.getsize(path)
        print(f"File size: {file_size} bytes")
        
        return file_size


def demo_mixed_data():
    """Demonstrate mixed RGB image and raw data with different codecs"""
    print("\n=== Mixed Data Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "mixed_demo.vla")
        
        # Create trajectory with default codec selection
        traj = Trajectory(path, mode="w", video_codec="auto")
        
        # Add mixed data
        for i in range(5):
            # RGB image - will use video codec
            rgb_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            
            # Non-image data - will use raw codec
            vector_data = np.random.rand(10).astype(np.float32)
            depth_data = np.random.rand(32, 32).astype(np.float32)  # Grayscale
            
            traj.add("camera/rgb", rgb_image, timestamp=i*100)
            traj.add("sensor/vector", vector_data, timestamp=i*100)
            traj.add("camera/depth", depth_data, timestamp=i*100)
        
        traj.close()
        
        # Read back and verify
        traj_read = Trajectory(path, mode="r")
        data = traj_read.load()
        traj_read.close()
        
        print(f"Loaded {len(data)} features:")
        for key, values in data.items():
            print(f"  {key}: shape={values.shape}, dtype={values.dtype}")
        
        file_size = os.path.getsize(path)
        print(f"File size: {file_size} bytes")
        
        return file_size


def demo_codec_config():
    """Demonstrate custom codec configuration"""
    print("\n=== Custom Codec Configuration Demo ===")
    
    # Create custom codec config
    config = CodecConfig(codec="rawvideo_pyarrow", options={
        "batch_size": 50,  # Smaller batches
        "compression": "lz4"  # Different compression
    })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "custom_config_demo.vla")
        
        # Create trajectory with custom config
        traj = Trajectory(path, mode="w", codec_config=config)
        
        # Add test data
        for i in range(20):
            vector_data = np.random.rand(8).astype(np.float32)
            traj.add("sensor/data", vector_data, timestamp=i*50)
        
        traj.close()
        
        # Read back and verify
        traj_read = Trajectory(path, mode="r")
        data = traj_read.load()
        traj_read.close()
        
        print(f"Loaded {len(data)} features:")
        for key, values in data.items():
            print(f"  {key}: shape={values.shape}, dtype={values.dtype}")
        
        file_size = os.path.getsize(path)
        print(f"File size: {file_size} bytes")
        
        return file_size


if __name__ == "__main__":
    print("Codec Abstraction System Demo")
    print("=" * 50)
    
    pickle_size = demo_pickle_codec()
    pyarrow_size = demo_pyarrow_codec()
    mixed_size = demo_mixed_data()
    custom_size = demo_codec_config()
    
    print("\n=== Summary ===")
    print(f"Pickle codec file size: {pickle_size} bytes")
    if pyarrow_size is not None:
        print(f"PyArrow codec file size: {pyarrow_size} bytes")
        if pickle_size:
            compression_ratio = pickle_size / pyarrow_size
            print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Mixed data file size: {mixed_size} bytes")
    print(f"Custom config file size: {custom_size} bytes")
    
    print("\nDemo completed successfully!") 