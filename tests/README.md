# robodm Test Suite

This directory contains comprehensive tests for the robodm trajectory management system, including unit tests, integration tests, and performance benchmarks.

## Test Structure

```
tests/
├── conftest.py           # Pytest configuration and fixtures
├── test_fixtures.py      # Mock objects and test data fixtures
├── test_trajectory.py    # Unit tests for trajectory functionality
├── test_loaders.py       # Unit tests for data loaders
├── test_benchmark.py     # Performance benchmarks
└── README.md            # This file
```

## Test Categories

### Unit Tests (`test_trajectory.py`, `test_loaders.py`)
- Test individual components in isolation
- Use mock dependencies for fast, deterministic testing
- Cover core functionality like data creation, loading, and manipulation

### Integration Tests
- Test complete workflows across multiple components
- Use real file system operations
- Verify end-to-end functionality

### Benchmark Tests (`test_benchmark.py`)
- Compare performance across different formats (VLA, HDF5, TFRecord)
- Measure file sizes, creation time, and loading time
- Generate CSV reports for analysis

## Running Tests

### Quick Start
```bash
# Run basic unit tests
python run_tests.py --test-type unit

# Run with coverage
python run_tests.py --test-type unit --coverage

# Run benchmarks
python run_tests.py --test-type benchmark --benchmark-size small
```

### Detailed Commands

#### Unit Tests Only
```bash
python -m pytest tests/test_trajectory.py tests/test_loaders.py -m "not slow"
```

#### Integration Tests
```bash
python -m pytest tests/ -m "integration"
```

#### Benchmark Tests
```bash
# Small benchmarks (fast)
python -m pytest tests/test_benchmark.py -m "not slow"

# Full benchmarks (slow)
python -m pytest tests/test_benchmark.py
```

#### All Tests
```bash
python -m pytest tests/
```

## Test Fixtures and Mock Objects

### MockFileSystem
- Simulates file system operations without actual I/O
- Enables fast, deterministic testing
- Located in `test_fixtures.py`

### MockTimeProvider
- Provides controllable time for testing
- Enables deterministic timestamp testing
- Supports time advancement simulation

### Sample Data
- `sample_trajectory_data`: Small datasets for quick tests
- `large_sample_data`: Larger datasets for performance testing
- `sample_dict_of_lists`: Test data in dictionary-of-lists format

## Benchmarking

The benchmark suite compares robodm VLA format against:
- **HDF5**: Popular scientific data format
- **TFRecord**: TensorFlow's native format (if available)

### Benchmark Metrics
- **File Size**: Compressed size on disk
- **Creation Time**: Time to write data to format
- **Loading Time**: Time to read data from format
- **Compression Ratio**: Uncompressed size / compressed size
- **Scalability**: Performance vs. dataset size

### Sample Benchmark Output
```
Format     | File Size (MB) | Creation (s) | Loading (s) | Compression
-----------|----------------|--------------|-------------|------------
VLA_lossy  | 12.34         | 2.15         | 0.89        | 8.2x
VLA_lossless| 18.67        | 1.98         | 0.76        | 5.4x
HDF5       | 15.23         | 1.87         | 0.92        | 6.6x
TFDS       | 20.45         | 3.21         | 1.23        | 4.9x
```

## Test Configuration

### Pytest Markers
- `@pytest.mark.slow`: Tests that take significant time
- `@pytest.mark.integration`: Integration tests requiring real I/O
- `@pytest.mark.benchmark`: Performance benchmark tests

### Environment Variables
```bash
# Skip slow tests
export PYTEST_IGNORE_SLOW=1

# Set custom temp directory
export PYTEST_TEMP_DIR=/path/to/temp

# Enable verbose logging
export ROBODM_TEST_VERBOSE=1
```

## Adding New Tests

### Unit Test Example
```python
def test_my_feature(temp_dir, mock_filesystem):
    """Test my new feature."""
    # Create test data
    data = {"feature": [1, 2, 3]}
    
    # Use mock filesystem for fast testing
    trajectory = Trajectory("test.vla", mode="w", filesystem=mock_filesystem)
    # ... test logic
    
    assert expected_result == actual_result
```

### Benchmark Test Example
```python
def test_my_benchmark(temp_dir):
    """Benchmark my feature."""
    runner = BenchmarkRunner(temp_dir)
    
    # Create test data
    data = runner.create_test_data(num_samples=100)
    
    # Run benchmarks
    results = runner.run_comprehensive_benchmark()
    
    # Verify results
    for result in results:
        assert result.compression_ratio > 1.0
```

## Dependencies

### Required
- `pytest`: Test framework
- `numpy`: Numerical operations
- `h5py`: HDF5 support

### Optional
- `tensorflow`: For TFRecord benchmarks
- `pytest-cov`: Coverage reporting
- `pytest-html`: HTML test reports
- `pytest-benchmark`: Advanced benchmarking

## Troubleshooting

### Common Issues

#### "TensorFlow not available"
```bash
pip install tensorflow  # For TFRecord benchmarks
```

#### "Permission denied" errors
```bash
# Ensure temp directory is writable
chmod 755 /tmp/robodm_tests
```

#### Out of memory errors
```bash
# Reduce benchmark dataset size
python run_tests.py --test-type benchmark --benchmark-size small
```

### Debug Mode
```bash
# Run with full output and debugging
python -m pytest tests/ -v -s --tb=long --no-header
```

## Performance Expectations

### Unit Tests
- Should complete in < 30 seconds
- Use minimal memory (< 100MB)
- No external dependencies

### Integration Tests
- Should complete in < 2 minutes
- May use up to 1GB memory
- Require file system access

### Benchmark Tests
- Small: < 1 minute, < 500MB memory
- Medium: < 5 minutes, < 2GB memory  
- Large: < 15 minutes, < 8GB memory

## Contributing

When adding new tests:

1. **Follow naming conventions**: `test_*.py` files, `test_*` functions
2. **Use appropriate fixtures**: Mock objects for unit tests, real I/O for integration
3. **Add appropriate markers**: `@pytest.mark.slow` for long-running tests
4. **Document test purpose**: Clear docstrings explaining what is tested
5. **Keep tests focused**: One concept per test function
6. **Use assertions effectively**: Clear, specific assertion messages

For benchmark tests:
1. **Use deterministic data**: Set random seeds for reproducible results
2. **Measure what matters**: Focus on user-relevant metrics
3. **Consider scalability**: Test with multiple dataset sizes
4. **Save results**: Generate CSV/reports for analysis 