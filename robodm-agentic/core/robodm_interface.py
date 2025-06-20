"""RoboDM interface for accessing trajectory data and metadata."""

from typing import List, Dict, Any, Iterator, Optional, Union
import os
import sys
import logging

# Add the parent robodm package to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import robodm
from robodm import Trajectory
from robodm.feature import FeatureType

logger = logging.getLogger(__name__)


class RoboDMInterface:
    """Interface for interacting with RoboDM database and trajectories."""
    
    def __init__(self, data_path: str, pattern: str = "*.vla"):
        """Initialize the RoboDM interface.
        
        Args:
            data_path: Path to directory containing trajectory files or single file path
            pattern: File pattern to match when data_path is a directory
        """
        self.data_path = data_path
        self.pattern = pattern
        self._trajectory_cache = {}
        
        # Discover trajectory files
        self.trajectory_files = self._discover_trajectories()
        logger.info(f"Discovered {len(self.trajectory_files)} trajectories")
    
    def _discover_trajectories(self) -> Dict[str, str]:
        """Discover all trajectory files in the data path."""
        trajectory_files = {}
        
        if os.path.isfile(self.data_path):
            # Single file
            trajectory_id = os.path.basename(self.data_path).replace('.vla', '')
            trajectory_files[trajectory_id] = self.data_path
        elif os.path.isdir(self.data_path):
            # Directory - find all matching files
            import glob
            files = glob.glob(os.path.join(self.data_path, self.pattern))
            for file_path in files:
                trajectory_id = os.path.basename(file_path).replace('.vla', '')
                trajectory_files[trajectory_id] = file_path
        else:
            logger.warning(f"Data path does not exist: {self.data_path}")
            
        return trajectory_files
    
    def get_available_functions(self) -> Dict[str, str]:
        """Return available functions with descriptions for LLM code generation."""
        return {
            "get_all_trajectories": "Get list of all trajectory IDs in the database",
            "get_trajectory_metadata": "Get metadata for a specific trajectory by ID",
            "get_trajectory_data": "Load complete trajectory data including all features",
            "get_trajectory_frames": "Get visual frames from a trajectory (if available)",
            "get_frame_by_index": "Get a specific frame by trajectory ID and frame index",
            "filter_trajectories_by_metadata": "Filter trajectories by metadata criteria",
            "search_trajectories": "Search trajectories by various criteria",
            "get_trajectory_status": "Get success/failure status of trajectory",
            "count_trajectories": "Count trajectories matching criteria",
            "get_feature_names": "Get all feature names available in a trajectory",
            "get_trajectory_length": "Get the number of timesteps in a trajectory",
            "slice_trajectory": "Get a subset of trajectory data using Python slice notation",
            "sample_trajectories": "Randomly sample N trajectories from the database"
        }
    
    def get_all_trajectories(self) -> List[str]:
        """Get all trajectory IDs."""
        return list(self.trajectory_files.keys())
    
    def get_trajectory_metadata(self, traj_id: str) -> Dict[str, Any]:
        """Get metadata for a trajectory."""
        if traj_id not in self.trajectory_files:
            raise ValueError(f"Trajectory {traj_id} not found")
            
        file_path = self.trajectory_files[traj_id]
        
        # Basic file metadata
        metadata = {
            "trajectory_id": traj_id,
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "modified_time": os.path.getmtime(file_path)
        }
        
        try:
            # Load trajectory to get additional metadata
            traj = self._get_trajectory(traj_id)
            traj_length = len(traj)
            feature_names = []
            if traj_length > 0:
                loaded_data = traj.load()
                if isinstance(loaded_data, dict):
                    feature_names = list(loaded_data.keys())
            
            metadata.update({
                "length": traj_length,
                "feature_names": feature_names,
                "status": "success"  # If we can load it, assume it's successful
            })
        except Exception as e:
            metadata.update({
                "status": "failed",
                "error": str(e)
            })
            
        return metadata
    
    def get_trajectory_data(self, traj_id: str, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Load complete trajectory data."""
        traj = self._get_trajectory(traj_id)
        data = traj.load()
        
        # Ensure data is a dictionary
        if not isinstance(data, dict):
            raise ValueError(f"Trajectory data is not in expected format for {traj_id}")
        
        if features:
            # Filter to requested features only
            filtered_data = {k: v for k, v in data.items() if k in features}
            return filtered_data
        
        return data
    
    def get_trajectory_frames(self, traj_id: str) -> List[Any]:
        """Get visual frames from a trajectory."""
        data = self.get_trajectory_data(traj_id)
        frames = []
        
        # Look for common image feature names
        image_features = [k for k in data.keys() if any(
            img_key in k.lower() for img_key in ['image', 'rgb', 'camera', 'vision', 'observation/image']
        )]
        
        if image_features:
            # Return frames from the first image feature found
            frames = data[image_features[0]]
            if not isinstance(frames, list):
                frames = list(frames)
                
        return frames
    
    def get_frame_by_index(self, traj_id: str, frame_idx: int) -> Any:
        """Get specific frame by index."""
        frames = self.get_trajectory_frames(traj_id)
        if frame_idx >= len(frames):
            raise IndexError(f"Frame index {frame_idx} out of range for trajectory {traj_id}")
        return frames[frame_idx]
    
    def filter_trajectories_by_metadata(self, criteria: Dict[str, Any]) -> List[str]:
        """Filter trajectories by metadata criteria."""
        matching_trajectories = []
        
        for traj_id in self.get_all_trajectories():
            try:
                metadata = self.get_trajectory_metadata(traj_id)
                
                # Check if all criteria are met
                matches = True
                for key, expected_value in criteria.items():
                    if key not in metadata:
                        matches = False
                        break
                    
                    actual_value = metadata[key]
                    
                    # Handle different comparison types
                    if isinstance(expected_value, str):
                        if actual_value != expected_value:
                            matches = False
                            break
                    elif isinstance(expected_value, (int, float)):
                        if actual_value != expected_value:
                            matches = False
                            break
                    elif isinstance(expected_value, dict):
                        # Handle range queries like {"min": 10, "max": 100}
                        if "min" in expected_value and actual_value < expected_value["min"]:
                            matches = False
                            break
                        if "max" in expected_value and actual_value > expected_value["max"]:
                            matches = False
                            break
                
                if matches:
                    matching_trajectories.append(traj_id)
                    
            except Exception as e:
                logger.warning(f"Error filtering trajectory {traj_id}: {e}")
                continue
                
        return matching_trajectories
    
    def search_trajectories(self, query: str) -> List[str]:
        """Search trajectories by text query in metadata."""
        matching_trajectories = []
        query_lower = query.lower()
        
        for traj_id in self.get_all_trajectories():
            try:
                metadata = self.get_trajectory_metadata(traj_id)
                
                # Search in trajectory ID and string metadata values
                searchable_text = traj_id.lower()
                for value in metadata.values():
                    if isinstance(value, str):
                        searchable_text += " " + value.lower()
                
                if query_lower in searchable_text:
                    matching_trajectories.append(traj_id)
                    
            except Exception as e:
                logger.warning(f"Error searching trajectory {traj_id}: {e}")
                continue
                
        return matching_trajectories
    
    def get_trajectory_status(self, traj_id: str) -> str:
        """Get success/failure status of trajectory."""
        metadata = self.get_trajectory_metadata(traj_id)
        return metadata.get('status', 'unknown')
    
    def count_trajectories(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """Count trajectories matching criteria."""
        if criteria:
            return len(self.filter_trajectories_by_metadata(criteria))
        return len(self.trajectory_files)
    
    def get_feature_names(self, traj_id: str) -> List[str]:
        """Get all feature names in a trajectory."""
        metadata = self.get_trajectory_metadata(traj_id)
        return metadata.get('feature_names', [])
    
    def get_trajectory_length(self, traj_id: str) -> int:
        """Get the number of timesteps in a trajectory."""
        traj = self._get_trajectory(traj_id)
        return len(traj)
    
    def slice_trajectory(self, traj_id: str, start: Optional[int] = None, 
                        end: Optional[int] = None, step: Optional[int] = None) -> Dict[str, Any]:
        """Get a subset of trajectory data using slice notation."""
        data = self.get_trajectory_data(traj_id)
        slice_obj = slice(start, end, step)
        
        sliced_data = {}
        for feature_name, feature_data in data.items():
            if hasattr(feature_data, '__getitem__'):
                sliced_data[feature_name] = feature_data[slice_obj]
            else:
                sliced_data[feature_name] = feature_data
                
        return sliced_data
    
    def sample_trajectories(self, n: int, seed: Optional[int] = None) -> List[str]:
        """Randomly sample N trajectories from the database."""
        import random
        if seed is not None:
            random.seed(seed)
            
        all_trajectories = self.get_all_trajectories()
        if n >= len(all_trajectories):
            return all_trajectories
            
        return random.sample(all_trajectories, n)
    
    def _get_trajectory(self, traj_id: str) -> Trajectory:
        """Get trajectory object with caching."""
        if traj_id not in self._trajectory_cache:
            if traj_id not in self.trajectory_files:
                raise ValueError(f"Trajectory {traj_id} not found")
                
            file_path = self.trajectory_files[traj_id]
            self._trajectory_cache[traj_id] = robodm.Trajectory(file_path, mode="r")
            
        return self._trajectory_cache[traj_id]
    
    def close_all(self):
        """Close all cached trajectory objects."""
        for traj in self._trajectory_cache.values():
            try:
                traj.close()
            except Exception as e:
                logger.warning(f"Error closing trajectory: {e}")
        self._trajectory_cache.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
