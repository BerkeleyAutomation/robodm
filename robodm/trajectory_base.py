from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Text, Union

import numpy as np


class TrajectoryInterface(ABC):
    """
    Abstract base class defining the interface for trajectory objects.
    This allows for better testing and dependency injection.
    """

    @abstractmethod
    def add(self,
            feature: str,
            data: Any,
            timestamp: Optional[int] = None,
            time_unit: Optional[str] = None) -> None:
        """Add a single feature value to the trajectory.
        
        Args:
            feature (str): name of the feature
            data (Any): value associated with the feature; except dictionary
            timestamp (optional int): timestamp value. If not provided, the current time is used.
            time_unit (optional str): time unit of the timestamp. If not provided, uses trajectory default.
        """
        pass

    @abstractmethod
    def add_by_dict(self,
                    data: Dict[str, Any],
                    timestamp: Optional[int] = None,
                    time_unit: Optional[str] = None) -> None:
        """Add multiple features from a dictionary to the trajectory.
        
        Args:
            data (Dict[str, Any]): dictionary of feature name and value
            timestamp (optional int): timestamp value. If not provided, the current time is used.
            time_unit (optional str): time unit of the timestamp. If not provided, uses trajectory default.
        """
        pass

    @abstractmethod
    def load(self,
             return_type: str = "numpy",
             desired_frequency: Optional[float] = None,
             data_slice: Optional[slice] = None) -> Union[Dict, Any]:
        """Load trajectory data with optional temporal resampling and slicing.
        
        Parameters
        ----------
        return_type : {"numpy", "container"}, default "numpy"
            • "numpy"     – decode the data and return a dict[str, np.ndarray]
            • "container" – skip all decoding and just return the file path
        desired_frequency : float | None, default None
            Target sampling frequency **in hertz**.  If None, every frame is
            returned (subject to `data_slice`).
        data_slice : slice | None, default None
            Standard Python slice that is applied *after* resampling.
        """
        pass

    @abstractmethod
    def close(self, compact: bool = True) -> None:
        """Close the trajectory file.
        
        Args:
            compact: re-read from the cache to encode pickled data to images
        """
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> Any:
        """Get a feature from the trajectory."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get the length of the trajectory."""
        pass

    @abstractmethod
    def init_feature_streams(self, feature_spec: Dict) -> None:
        """Initialize the feature stream with the feature name and its type.
        
        Args:
            feature_spec: dictionary of feature name and its type
        """
        pass

    @classmethod
    @abstractmethod
    def from_list_of_dicts(
        cls,
        data: List[Dict[str, Any]],
        path: Text,
        video_codec: str = "auto",
        codec_options: Optional[Dict[str, Any]] = None,
        visualization_feature: Optional[Text] = None,
        fps: Optional[int] = 10,
    ) -> "TrajectoryInterface":
        """
        Create a Trajectory object from a list of dictionaries.

        Args:
            data (List[Dict[str, Any]]): list of dictionaries
            path (Text): path to the trajectory file
            video_codec (str, optional): Video codec to use. Defaults to "auto".
            codec_options (Dict[str, Any], optional): Additional codec-specific options.
            visualization_feature: Optional feature name to prioritize as first stream for visualization.
            fps: Optional frames per second for timestamp calculation.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict_of_lists(
        cls,
        data: Dict[str, List[Any]],
        path: Text,
        feature_name_separator: Text = "/",
        video_codec: str = "auto",
        codec_options: Optional[Dict[str, Any]] = None,
        visualization_feature: Optional[Text] = None,
        fps: Optional[int] = 10,
    ) -> "TrajectoryInterface":
        """
        Create a Trajectory object from a dictionary of lists.

        Args:
            data (Dict[str, List[Any]]): dictionary of lists. Assume list length is the same for all features.
            path (Text): path to the trajectory file
            feature_name_separator (Text, optional): Delimiter to separate feature names. Defaults to "/".
            video_codec (str, optional): Video codec to use. Defaults to "auto".
            codec_options (Dict[str, Any], optional): Additional codec-specific options.
            visualization_feature: Optional feature name to prioritize as first stream for visualization.
            fps: Optional frames per second for timestamp calculation.
        """
        pass


class FileSystemInterface(ABC):
    """Abstract interface for file system operations to enable testing with mocks."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file exists."""
        pass

    @abstractmethod
    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        """Create directories."""
        pass

    @abstractmethod
    def remove(self, path: str) -> None:
        """Remove a file."""
        pass

    @abstractmethod
    def rename(self, src: str, dst: str) -> None:
        """Rename a file."""
        pass


class DefaultFileSystem(FileSystemInterface):
    """Default implementation using standard file system operations."""

    def exists(self, path: str) -> bool:
        import os

        return os.path.exists(path)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        import os

        os.makedirs(path, exist_ok=exist_ok)

    def remove(self, path: str) -> None:
        import os

        os.remove(path)

    def rename(self, src: str, dst: str) -> None:
        import os

        os.rename(src, dst)


class TimeProvider(ABC):
    """Abstract interface for time operations to enable testing."""

    @abstractmethod
    def time(self) -> float:
        """Get current time."""
        pass


class DefaultTimeProvider(TimeProvider):
    """Default implementation using standard time operations."""

    def time(self) -> float:
        import time

        return time.time()
