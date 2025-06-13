from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class CodecPacket:
    """Container-agnostic representation of encoded data"""
    data: bytes
    metadata: Dict[str, Any]  # Codec-specific metadata
    seekable: bool = False  # Whether this packet can be used for seeking


class DataCodec(ABC):
    """Abstract base class for data codecs"""
    
    @abstractmethod
    def encode(self, data: Any, timestamp: int, **kwargs) -> List[CodecPacket]:
        """Encode data into codec packets
        
        Args:
            data: The data to encode
            timestamp: Timestamp in milliseconds
            **kwargs: Additional codec-specific parameters
            
        Returns:
            List of CodecPacket objects
        """
        pass
    
    @abstractmethod
    def decode(self, packet: CodecPacket) -> Any:
        """Decode a codec packet back to original data
        
        Args:
            packet: CodecPacket to decode
            
        Returns:
            Decoded data
        """
        pass
    
    @abstractmethod
    def flush(self) -> List[CodecPacket]:
        """Flush any buffered data
        
        Returns:
            List of remaining CodecPacket objects
        """
        pass
    
    @abstractmethod
    def supports_seeking(self) -> bool:
        """Whether this codec supports efficient seeking"""
        pass
    
    @abstractmethod
    def get_codec_name(self) -> str:
        """Get the codec identifier name"""
        pass


class VideoCodec(DataCodec):
    """Abstract base class for video codecs (like H.264, FFV1, etc.)"""
    
    @abstractmethod
    def configure_stream(self, stream: Any, feature_type: Any) -> None:
        """Configure a container stream for this video codec
        
        Args:
            stream: Backend-specific stream object
            feature_type: FeatureType object with shape information
        """
        pass
    
    @abstractmethod
    def create_frame(self, data: np.ndarray, timestamp: int) -> Any:
        """Create a backend-specific frame object
        
        Args:
            data: Image data as numpy array
            timestamp: Timestamp in milliseconds
            
        Returns:
            Backend-specific frame object
        """
        pass


class RawDataCodec(DataCodec):
    """Abstract base class for raw data codecs (for non-image data)"""
    
    @abstractmethod
    def get_container_encoding(self) -> str:
        """Get the container-level encoding string to use"""
        pass 