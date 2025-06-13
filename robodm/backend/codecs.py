"""Concrete implementations of data codecs"""

import pickle
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from .codec_interface import DataCodec, CodecPacket, RawDataCodec, VideoCodec

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import io
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logger.warning("PyArrow not available - PyArrowRawCodec will not work")


class PickleRawCodec(RawDataCodec):
    """Pickle-based codec for raw data (current default behavior)"""
    
    def __init__(self):
        self.codec_name = "pickle_raw"
    
    def encode(self, data: Any, timestamp: int, **kwargs) -> List[CodecPacket]:
        """Encode data using pickle"""
        try:
            payload = pickle.dumps(data)
            packet = CodecPacket(
                data=payload,
                metadata={
                    "pts": timestamp,
                    "dts": timestamp,
                    "codec": self.codec_name,
                    "original_type": type(data).__name__,
                    "data_shape": getattr(data, 'shape', None),
                    "data_dtype": str(getattr(data, 'dtype', None)) if hasattr(data, 'dtype') else None
                },
                seekable=False  # Individual pickled packets are not seekable
            )
            return [packet]
        except Exception as e:
            logger.error(f"Failed to pickle encode data: {e}")
            raise
    
    def decode(self, packet: CodecPacket) -> Any:
        """Decode pickled data"""
        try:
            return pickle.loads(packet.data)
        except Exception as e:
            logger.error(f"Failed to pickle decode data: {e}")
            raise
    
    def flush(self) -> List[CodecPacket]:
        """No buffering in pickle codec"""
        return []
    
    def supports_seeking(self) -> bool:
        """Pickle codec doesn't support seeking"""
        return False
    
    def get_codec_name(self) -> str:
        return self.codec_name
    
    def get_container_encoding(self) -> str:
        return "rawvideo"


class PyArrowBatchCodec(RawDataCodec):
    """PyArrow-based codec that batches data for better seeking"""
    
    def __init__(self, batch_size: int = 100, compression: str = "snappy"):
        if not PYARROW_AVAILABLE:
            raise ImportError("PyArrow is required for PyArrowBatchCodec")
        
        self.codec_name = "pyarrow_batch"
        self.batch_size = batch_size
        self.compression = compression
        self.current_batch: List[Dict[str, Any]] = []
        self.batch_start_timestamp: Optional[int] = None
    
    def encode(self, data: Any, timestamp: int, **kwargs) -> List[CodecPacket]:
        """Encode data using PyArrow batching"""
        try:
            # Convert numpy arrays to Python objects for Arrow compatibility
            if isinstance(data, np.ndarray):
                serialized_data = data.tobytes()
                data_info = {
                    "type": "numpy",
                    "shape": data.shape,
                    "dtype": str(data.dtype),
                    "data": serialized_data
                }
            else:
                # Fallback to pickle for complex objects
                data_info = {
                    "type": "pickle",
                    "data": pickle.dumps(data)
                }
            
            # Add to current batch
            entry = {
                "pts": timestamp,
                "dts": timestamp,
                "data_info": data_info
            }
            
            if self.batch_start_timestamp is None:
                self.batch_start_timestamp = timestamp
            
            self.current_batch.append(entry)
            
            # Check if batch is full
            if len(self.current_batch) >= self.batch_size:
                return self._flush_batch()
            
            return []  # No packets yet
            
        except Exception as e:
            logger.error(f"Failed to encode data with PyArrow: {e}")
            raise
    
    def _flush_batch(self) -> List[CodecPacket]:
        """Flush the current batch to a packet"""
        if not self.current_batch:
            return []
        
        try:
            # Create Arrow table from batch
            table = pa.table({
                "pts": [entry["pts"] for entry in self.current_batch],
                "dts": [entry["dts"] for entry in self.current_batch], 
                "data_type": [entry["data_info"]["type"] for entry in self.current_batch],
                "data_shape": [entry["data_info"].get("shape") for entry in self.current_batch],
                "data_dtype": [entry["data_info"].get("dtype") for entry in self.current_batch],
                "data_bytes": [entry["data_info"]["data"] for entry in self.current_batch]
            })
            
            # Serialize to parquet in memory
            buffer = io.BytesIO()
            pq.write_table(table, buffer, compression=self.compression)
            payload = buffer.getvalue()
            
            batch_start = self.batch_start_timestamp
            batch_end = self.current_batch[-1]["pts"]
            
            packet = CodecPacket(
                data=payload,
                metadata={
                    "codec": self.codec_name,
                    "batch_start_pts": batch_start,
                    "batch_end_pts": batch_end,
                    "batch_size": len(self.current_batch),
                    "compression": self.compression
                },
                seekable=True  # Batched data supports seeking
            )
            
            # Reset batch
            self.current_batch = []
            self.batch_start_timestamp = None
            
            return [packet]
            
        except Exception as e:
            logger.error(f"Failed to flush PyArrow batch: {e}")
            raise
    
    def decode(self, packet: CodecPacket) -> List[Any]:
        """Decode PyArrow batch packet to list of data items"""
        try:
            buffer = io.BytesIO(packet.data)
            table = pq.read_table(buffer)
            
            # Convert back to original data
            results = []
            for i in range(len(table)):
                row = table.slice(i, 1)
                data_type = row["data_type"][0].as_py()
                data_bytes = row["data_bytes"][0].as_py()
                pts = row["pts"][0].as_py()
                
                if data_type == "numpy":
                    shape = row["data_shape"][0].as_py()
                    dtype = row["data_dtype"][0].as_py() 
                    data = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
                else:  # pickle
                    data = pickle.loads(data_bytes)
                
                results.append((pts, data))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to decode PyArrow batch: {e}")
            raise
    
    def flush(self) -> List[CodecPacket]:
        """Flush any remaining batched data"""
        return self._flush_batch()
    
    def supports_seeking(self) -> bool:
        """PyArrow codec supports seeking within batches"""
        return True
    
    def get_codec_name(self) -> str:
        return self.codec_name
    
    def get_container_encoding(self) -> str:
        return "rawvideo"


class PyAVVideoCodec(VideoCodec):
    """PyAV-based video codec wrapper"""
    
    def __init__(self, codec_name: str = None, **kwargs):
        # Handle both old and new initialization styles
        if codec_name is None:
            # New style: codec name should be passed as kwarg or inferred from registration
            self.codec_name = kwargs.get('codec_name', 'libx264')
            self.codec_config = kwargs
        else:
            # Old style: codec_name and codec_config passed separately
            self.codec_name = codec_name
            self.codec_config = kwargs.get('codec_config', kwargs)
        
        self._stream = None
    
    def configure_stream(self, stream: Any, feature_type: Any) -> None:
        """Configure PyAV stream for video codec"""
        self._stream = stream
        
        # Configure video codec settings
        if hasattr(feature_type, 'shape') and feature_type.shape:
            shape = feature_type.shape
            if len(shape) >= 2:
                stream.width = shape[1]
                stream.height = shape[0]
        
        # Set pixel format
        pixel_fmt = self.codec_config.get("pixel_format")
        if pixel_fmt:
            stream.pix_fmt = pixel_fmt
        
        # Set codec options
        codec_opts = self.codec_config.get("options", {})
        if codec_opts:
            stream.codec_context.options = codec_opts
    
    def create_frame(self, data: np.ndarray, timestamp: int) -> Any:
        """Create PyAV frame from image data"""
        import av
        
        # Convert to uint8 if needed
        if data.dtype == np.float32:
            data = np.clip(data * 255, 0, 255).astype(np.uint8)
        elif data.dtype != np.uint8:
            if np.issubdtype(data.dtype, np.integer):
                data = np.clip(data, 0, 255).astype(np.uint8)
            else:
                data = np.clip(data * 255, 0, 255).astype(np.uint8)
        
        # Only handle RGB images (HxWx3)
        if len(data.shape) != 3 or data.shape[2] != 3:
            raise ValueError(
                "Video codecs only support RGB images with shape (H, W, 3). "
                f"Got shape {data.shape}."
            )
        
        # Create RGB frame and convert to YUV420p when required
        if self.codec_name in {"libaom-av1", "ffv1", "libx264", "libx265"}:
            frame = av.VideoFrame.from_ndarray(data, format="rgb24")
            frame = frame.reformat(format="yuv420p")
        else:
            frame = av.VideoFrame.from_ndarray(data, format="rgb24")
        
        frame.pts = timestamp
        frame.dts = timestamp
        
        return frame
    
    def encode(self, data: Any, timestamp: int, **kwargs) -> List[CodecPacket]:
        """Encode video frame"""
        if self._stream is None:
            raise RuntimeError("Stream not configured")
        
        try:
            frame = self.create_frame(data, timestamp)
            packets = []
            
            # Encode frame to packets
            for pkt in self._stream.encode(frame):
                codec_packet = CodecPacket(
                    data=bytes(pkt),
                    metadata={
                        "pts": pkt.pts,
                        "dts": pkt.dts,
                        "codec": self.codec_name,
                        "is_keyframe": bool(getattr(pkt, 'is_keyframe', False))
                    },
                    seekable=bool(getattr(pkt, 'is_keyframe', False))
                )
                packets.append(codec_packet)
            
            return packets
            
        except Exception as e:
            logger.error(f"Failed to encode video frame: {e}")
            raise
    
    def decode(self, packet: CodecPacket) -> Any:
        """Decode video packet - delegated to container backend"""
        # Video decoding is handled by the container backend
        # This method is here for interface completeness
        raise NotImplementedError("Video decoding is handled by container backend")
    
    def flush(self) -> List[CodecPacket]:
        """Flush video encoder"""
        if self._stream is None:
            return []
        
        try:
            packets = []
            for pkt in self._stream.encode(None):
                codec_packet = CodecPacket(
                    data=bytes(pkt),
                    metadata={
                        "pts": pkt.pts,
                        "dts": pkt.dts,
                        "codec": self.codec_name,
                        "is_keyframe": bool(getattr(pkt, 'is_keyframe', False))
                    },
                    seekable=bool(getattr(pkt, 'is_keyframe', False))
                )
                packets.append(codec_packet)
            return packets
        except Exception:
            return []
    
    def supports_seeking(self) -> bool:
        """Video codecs support seeking to keyframes"""
        return True
    
    def get_codec_name(self) -> str:
        return self.codec_name


# Codec factory registry
_codec_factories: Dict[str, type] = {}
_codec_instances: Dict[str, DataCodec] = {}


def register_codec(name: str, codec_class: type):
    """Register a codec class with the factory"""
    if not issubclass(codec_class, DataCodec):
        raise TypeError(f"Codec class must inherit from DataCodec, got {codec_class}")
    _codec_factories[name] = codec_class


def get_codec(codec_name: str, **kwargs) -> DataCodec:
    """Get or create a codec instance"""
    cache_key = f"{codec_name}_{hash(str(sorted(kwargs.items())))}"
    
    if cache_key not in _codec_instances:
        if codec_name not in _codec_factories:
            raise ValueError(f"Unknown codec: {codec_name}. Available: {list(_codec_factories.keys())}")
        
        codec_class = _codec_factories[codec_name]
        _codec_instances[cache_key] = codec_class(**kwargs)
    
    return _codec_instances[cache_key]


def list_available_codecs() -> List[str]:
    """List all available codec names"""
    return list(_codec_factories.keys())


def clear_codec_cache():
    """Clear the codec registry cache"""
    global _codec_instances
    _codec_instances.clear()


def is_video_codec(codec_name: str) -> bool:
    """Check if a codec is a video codec"""
    if codec_name not in _codec_factories:
        return False
    return issubclass(_codec_factories[codec_name], VideoCodec)


def is_raw_codec(codec_name: str) -> bool:
    """Check if a codec is a raw data codec"""
    if codec_name not in _codec_factories:
        return False
    return issubclass(_codec_factories[codec_name], RawDataCodec)


# Register built-in codecs
register_codec("pickle_raw", PickleRawCodec)
if PYARROW_AVAILABLE:
    register_codec("pyarrow_batch", PyArrowBatchCodec)
register_codec("ffv1", PyAVVideoCodec)
register_codec("libaom-av1", PyAVVideoCodec)
register_codec("libx264", PyAVVideoCodec)
register_codec("libx265", PyAVVideoCodec) 