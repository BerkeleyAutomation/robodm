import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text, Union

import numpy as np

try:
    import ray
    import ray.data as rd

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from robodm.loader.vla import (LoadingMode, RayVLALoader, SliceConfig,
                               create_slice_loader, create_trajectory_loader)
from robodm.utils.flatten import data_to_tf_schema


@dataclass
class DatasetConfig:
    """Configuration for VLADataset."""

    batch_size: int = 1
    shuffle: bool = False
    num_parallel_reads: int = 4
    ray_init_kwargs: Optional[Dict] = None


class VLADataset:
    """
    Ray Dataset-based VLA dataset supporting both trajectory and slice loading modes.

    This dataset provides:
    1. Parallel data loading using Ray Dataset
    2. Automatic shuffling and splitting
    3. Support for both trajectory and slice loading modes
    4. Efficient data management for large datasets
    """

    def __init__(
        self,
        path: Text,
        mode: Union[str, LoadingMode] = LoadingMode.TRAJECTORY,
        split: str = "all",
        return_type: str = "numpy",
        config: Optional[DatasetConfig] = None,
        slice_config: Optional[SliceConfig] = None,
        **kwargs,
    ):
        """
        Initialize VLA dataset.

        Args:
            path: Path to VLA files (can be glob pattern, directory, or single file)
            mode: Loading mode ("trajectory" or "slice", or LoadingMode enum)
            split: Data split ("all", "train", "val")
            return_type: Return type ("numpy", "tensor", "container")
            config: Dataset configuration
            slice_config: Slice configuration (required if mode="slice")
            **kwargs: Additional arguments passed to RayVLALoader
        """
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray is required for VLADataset. Install with: pip install 'ray[data]'"
            )

        self.path = path
        self.return_type = return_type
        self.config = config or DatasetConfig()

        # Handle string mode input
        if isinstance(mode, str):
            mode = LoadingMode.TRAJECTORY if mode == "trajectory" else LoadingMode.SLICE
        self.mode = mode

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(**(self.config.ray_init_kwargs or {}))

        # Create the loader
        self.loader = RayVLALoader(
            path=path,
            mode=mode,
            batch_size=self.config.batch_size,
            return_type=return_type,
            shuffle=self.config.shuffle,
            num_parallel_reads=self.config.num_parallel_reads,
            slice_config=slice_config,
            **kwargs,
        )

        # Cache for schema and stats
        self._schema = None
        self._stats = None

    @classmethod
    def create_trajectory_dataset(
        cls,
        path: Text,
        split: str = "all",
        return_type: str = "numpy",
        config: Optional[DatasetConfig] = None,
        **kwargs,
    ) -> "VLADataset":
        """Create a dataset for loading complete trajectories."""
        return cls(
            path=path,
            mode=LoadingMode.TRAJECTORY,
            return_type=return_type,
            config=config,
            **kwargs,
        )

    @classmethod
    def create_slice_dataset(
        cls,
        path: Text,
        slice_length: int = 100,
        return_type: str = "numpy",
        config: Optional[DatasetConfig] = None,
        min_slice_length: Optional[int] = None,
        stride: int = 1,
        random_start: bool = True,
        overlap_ratio: float = 0.0,
        **kwargs,
    ) -> "VLADataset":
        """Create a dataset for loading trajectory slices."""
        slice_config = SliceConfig(
            slice_length=slice_length,
            min_slice_length=min_slice_length,
            stride=stride,
            random_start=random_start,
            overlap_ratio=overlap_ratio,
        )

        return cls(
            path=path,
            mode=LoadingMode.SLICE,
            return_type=return_type,
            config=config,
            slice_config=slice_config,
            **kwargs,
        )

    def get_ray_dataset(self) -> rd.Dataset:
        """Get the underlying Ray dataset."""
        return self.loader.dataset

    def iter_batches(self, batch_size: Optional[int] = None):
        """Iterate over batches of data."""
        return self.loader.iter_batches(batch_size)

    def iter_rows(self):
        """Iterate over individual rows of data."""
        return self.loader.iter_rows()

    def take(self, num_items: int) -> List[Dict[str, Any]]:
        """Take a specific number of items."""
        return self.loader.take(num_items)

    def sample(self,
               num_samples: int,
               replace: bool = False) -> List[Dict[str, Any]]:
        """Sample from the dataset."""
        return list(self.loader.sample(num_samples, replace))

    def count(self) -> int:
        """Count the number of items in the dataset."""
        return self.loader.count()

    def schema(self):
        """Get the schema of the dataset."""
        if self._schema is None:
            self._schema = self.loader.schema()
        return self._schema

    def split(self, *fractions: float, shuffle: bool = True):
        """Split the dataset into multiple datasets."""
        ray_datasets = self.loader.split(*fractions, shuffle=shuffle)

        # Create new VLADataset instances for each split
        split_datasets = []
        for ray_ds in ray_datasets:
            split_dataset = VLADataset.__new__(VLADataset)
            split_dataset.path = self.path
            split_dataset.mode = self.mode
            split_dataset.return_type = self.return_type
            split_dataset.config = self.config
            split_dataset.loader = self.loader.__class__.__new__(
                self.loader.__class__)
            split_dataset.loader.dataset = ray_ds
            split_dataset._schema = self._schema
            split_dataset._stats = None
            split_datasets.append(split_dataset)

        return split_datasets

    def filter(self, fn):
        """Filter the dataset."""
        filtered_dataset = VLADataset.__new__(VLADataset)
        filtered_dataset.path = self.path
        filtered_dataset.mode = self.mode
        filtered_dataset.return_type = self.return_type
        filtered_dataset.config = self.config
        filtered_dataset.loader = self.loader.__class__.__new__(
            self.loader.__class__)
        filtered_dataset.loader.dataset = self.loader.dataset.filter(fn)
        filtered_dataset._schema = self._schema
        filtered_dataset._stats = None
        return filtered_dataset

    def map(self, fn, **kwargs):
        """Map a function over the dataset."""
        mapped_dataset = VLADataset.__new__(VLADataset)
        mapped_dataset.path = self.path
        mapped_dataset.mode = self.mode
        mapped_dataset.return_type = self.return_type
        mapped_dataset.config = self.config
        mapped_dataset.loader = self.loader.__class__.__new__(
            self.loader.__class__)
        mapped_dataset.loader.dataset = self.loader.dataset.map(fn, **kwargs)
        mapped_dataset._schema = None  # Schema might change after mapping
        mapped_dataset._stats = None
        return mapped_dataset

    def shuffle(self, seed: Optional[int] = None):
        """Shuffle the dataset."""
        shuffled_dataset = VLADataset.__new__(VLADataset)
        shuffled_dataset.path = self.path
        shuffled_dataset.mode = self.mode
        shuffled_dataset.return_type = self.return_type
        shuffled_dataset.config = self.config
        shuffled_dataset.loader = self.loader.__class__.__new__(
            self.loader.__class__)
        shuffled_dataset.loader.dataset = self.loader.dataset.random_shuffle(
            seed=seed)
        shuffled_dataset._schema = self._schema
        shuffled_dataset._stats = None
        return shuffled_dataset

    def materialize(self):
        """Materialize the dataset in memory."""
        return self.loader.materialize()

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if self._stats is None:
            sample = self.peek()
            if sample:
                self._stats = {
                    "mode":
                    self.mode.value,
                    "return_type":
                    self.return_type,
                    "total_items":
                    self.count(),
                    "sample_keys":
                    (list(sample.keys()) if isinstance(sample, dict) else []),
                }

                # Add mode-specific stats
                if self.mode == LoadingMode.TRAJECTORY:
                    # For trajectory mode, estimate length from first key
                    first_key = next(iter(sample.keys())) if sample else None
                    if first_key and hasattr(sample[first_key], "__len__"):
                        self._stats["trajectory_length"] = len(
                            sample[first_key])
                elif self.mode == LoadingMode.SLICE:
                    # For slice mode, estimate length from first key
                    first_key = next(iter(sample.keys())) if sample else None
                    if first_key and hasattr(sample[first_key], "__len__"):
                        self._stats["slice_length"] = len(sample[first_key])
                        self._stats["slice_start"] = (
                            0  # Cannot determine from direct data
                        )
                        self._stats["slice_end"] = len(sample[first_key])
            else:
                self._stats = {"mode": self.mode.value, "total_items": 0}

        return self._stats

    def peek(self) -> Optional[Dict[str, Any]]:
        """Peek at the first item without consuming it."""
        return self.loader.peek()

    def get_tf_schema(self):
        """Get TensorFlow schema for the dataset."""
        sample = self.peek()
        if sample:
            return data_to_tf_schema(sample)
        return None

    # Legacy compatibility methods
    def __iter__(self):
        """Iterate over the dataset (legacy compatibility)."""
        for item in self.loader.iter_rows():
            yield item

    def __next__(self):
        """Get next item (legacy compatibility)."""
        batch = self.loader.get_batch()
        if batch:
            return batch[0]
        raise StopIteration

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return self.count()

    def __getitem__(self, index):
        """Not supported for Ray datasets - use take() or sample() instead."""
        raise NotImplementedError(
            "Random access not supported for Ray datasets. "
            "Use take(), sample(), or iterate over the dataset instead.")

    def get_loader(self):
        """Get the underlying loader (legacy compatibility)."""
        return self.loader

    def get_next_trajectory(self):
        """Get next trajectory (legacy compatibility)."""
        item = next(self)
        return item


# Utility functions for common dataset operations
def load_trajectory_dataset(
    path: Text,
    split: str = "all",
    return_type: str = "numpy",
    batch_size: int = 1,
    shuffle: bool = False,
    num_parallel_reads: int = 4,
    **kwargs,
) -> VLADataset:
    """Load a dataset for complete trajectories."""
    config = DatasetConfig(batch_size=batch_size,
                           shuffle=shuffle,
                           num_parallel_reads=num_parallel_reads)
    return VLADataset.create_trajectory_dataset(path=path,
                                                return_type=return_type,
                                                config=config,
                                                **kwargs)


def load_slice_dataset(
    path: Text,
    slice_length: int = 100,
    split: str = "all",
    return_type: str = "numpy",
    batch_size: int = 1,
    shuffle: bool = False,
    num_parallel_reads: int = 4,
    min_slice_length: Optional[int] = None,
    stride: int = 1,
    random_start: bool = True,
    overlap_ratio: float = 0.0,
    **kwargs,
) -> VLADataset:
    """Load a dataset for trajectory slices."""
    config = DatasetConfig(batch_size=batch_size,
                           shuffle=shuffle,
                           num_parallel_reads=num_parallel_reads)
    return VLADataset.create_slice_dataset(
        path=path,
        slice_length=slice_length,
        return_type=return_type,
        config=config,
        min_slice_length=min_slice_length,
        stride=stride,
        random_start=random_start,
        overlap_ratio=overlap_ratio,
        **kwargs,
    )


def split_dataset(
    dataset: VLADataset,
    train_fraction: float = 0.8,
    val_fraction: float = 0.2,
    shuffle: bool = False,
) -> tuple[VLADataset, VLADataset]:
    """Split a dataset into train and validation sets."""
    if abs(train_fraction + val_fraction - 1.0) > 1e-6:
        raise ValueError("train_fraction + val_fraction must equal 1.0")

    splits = dataset.split(train_fraction, val_fraction, shuffle=shuffle)
    return splits[0], splits[1]
