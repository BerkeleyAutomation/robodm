"""
Codec Manager for handling codec instantiation and packet processing.

This provides an extensible way to manage codecs without case-by-case handling
in the backend implementations.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .codec_interface import DataCodec, RawDataCodec, VideoCodec, CodecPacket
from .codecs import get_codec, is_video_codec, is_raw_codec, list_available_codecs
from .base import PacketInfo

logger = logging.getLogger(__name__)


class CodecManager:
    """Manages codec instances and handles packet encoding/decoding"""
    
    def __init__(self):
        # Map stream_index -> codec instance
        self._stream_codecs: Dict[int, DataCodec] = {}
        # Map stream_index -> codec configuration
        self._stream_configs: Dict[int, Dict[str, Any]] = {}
    
    def create_codec_for_stream(
        self, 
        stream_index: int, 
        encoding: str, 
        codec_config: Any,
        feature_type: Any = None,
        stream: Any = None
    ) -> Optional[DataCodec]:
        """Create and configure a codec for a stream"""
        # Determine the actual codec to use
        raw_codec_name = self._determine_codec_name(encoding, codec_config)
        
        # Get codec configuration
        config = self._build_codec_config(raw_codec_name, codec_config, feature_type)
        
        # Create codec instance
        if is_video_codec(raw_codec_name):
            # For video codecs, pass codec_name in config if not already present
            if 'codec_name' not in config:
                config['codec_name'] = raw_codec_name
            codec = get_codec(raw_codec_name, **config)
        else:
            codec = get_codec(raw_codec_name, **config)
        
        # Configure the codec if needed
        if isinstance(codec, VideoCodec) and stream is not None:
            codec.configure_stream(stream, feature_type)
        
        # Cache the codec and its config
        self._stream_codecs[stream_index] = codec
        self._stream_configs[stream_index] = config
        
        logger.debug(f"Created codec {raw_codec_name} for stream {stream_index}")
        return codec
        

    
    def get_codec_for_stream(self, stream_index: int) -> Optional[DataCodec]:
        """Get the codec instance for a stream"""
        return self._stream_codecs.get(stream_index)
    
    def encode_data(
        self, 
        stream_index: int, 
        data: Any, 
        timestamp: int, 
        stream: Any = None
    ) -> List[PacketInfo]:
        """Encode data using the appropriate codec for the stream"""
        codec = self._stream_codecs.get(stream_index)
        if codec is None:
            logger.error(f"No codec found for stream {stream_index}")
            return []
        
        try:
            # Encode data to codec packets
            codec_packets = codec.encode(data, timestamp)
            
            # Convert to PacketInfo objects
            packet_infos = []
            for codec_packet in codec_packets:
                packet_info = self._codec_packet_to_packet_info(
                    codec_packet, stream_index, timestamp, stream
                )
                packet_infos.append(packet_info)
            
            return packet_infos
            
        except Exception as e:
            logger.error(f"Failed to encode data for stream {stream_index}: {e}")
            return []
    
    def flush_stream(self, stream_index: int, stream: Any = None) -> List[PacketInfo]:
        """Flush any buffered data from a stream's codec"""
        codec = self._stream_codecs.get(stream_index)
        if codec is None:
            return []
        
        try:
            codec_packets = codec.flush()
            packet_infos = []
            
            for codec_packet in codec_packets:
                packet_info = self._codec_packet_to_packet_info(
                    codec_packet, stream_index, None, stream
                )
                packet_infos.append(packet_info)
            
            return packet_infos
            
        except Exception as e:
            logger.error(f"Failed to flush stream {stream_index}: {e}")
            return []
    
    def decode_packet(self, packet_info: PacketInfo) -> Any:
        """Decode a packet using the appropriate codec"""
        stream_index = packet_info.stream_index
        codec = self._stream_codecs.get(stream_index)
        
        if codec is None:
            logger.warning(f"No codec found for stream {stream_index}, using fallback")
            return self._fallback_decode(packet_info)
        
        try:
            # Convert PacketInfo to CodecPacket
            codec_packet = self._packet_info_to_codec_packet(packet_info, codec)
            
            # Decode using codec
            return codec.decode(codec_packet)
            
        except Exception as e:
            logger.error(f"Failed to decode packet for stream {stream_index}: {e}")
            return self._fallback_decode(packet_info)
    
    def clear_stream_codecs(self):
        """Clear all stream codecs"""
        self._stream_codecs.clear()
        self._stream_configs.clear()
    
    def get_codec_info(self, stream_index: int) -> Optional[Dict[str, Any]]:
        """Get information about the codec for a stream"""
        codec = self._stream_codecs.get(stream_index)
        if codec is None:
            return None
        
        return {
            "codec_name": codec.get_codec_name(),
            "supports_seeking": codec.supports_seeking(),
            "is_video_codec": isinstance(codec, VideoCodec),
            "is_raw_codec": isinstance(codec, RawDataCodec),
            "config": self._stream_configs.get(stream_index, {})
        }
    
    # Private helper methods
    
    def _determine_codec_name(self, encoding: str, codec_config: Any) -> str:
        """Determine the actual codec name to use"""
        if encoding in {"ffv1", "libaom-av1", "libx264", "libx265"}:
            return encoding
        elif encoding.startswith("rawvideo"):
            # For rawvideo, check the codec config for the specific implementation
            if hasattr(codec_config, 'get_raw_codec_name'):
                return codec_config.get_raw_codec_name(encoding)
            else:
               raise ValueError(f"Unknown encoding {encoding}")
        else:
            raise ValueError(f"Unknown encoding {encoding}")
    
    def _build_codec_config(self, codec_name: str, codec_config: Any, feature_type: Any) -> Dict[str, Any]:
        """Build configuration dictionary for codec creation"""
        config = {}
        
        # Add codec name for video codecs that need it
        if is_video_codec(codec_name):
            # For video codecs, pass codec_name as first positional argument
            # and other config as keyword arguments
            if hasattr(codec_config, 'get_pixel_format'):
                pixel_fmt = codec_config.get_pixel_format(codec_name, feature_type)
                if pixel_fmt:
                    config["pixel_format"] = pixel_fmt
            
            if hasattr(codec_config, 'get_codec_options'):
                codec_opts = codec_config.get_codec_options(codec_name)
                if codec_opts:
                    config["options"] = codec_opts
        
        elif is_raw_codec(codec_name):
            # Add raw codec specific config
            if hasattr(codec_config, 'get_codec_options'):
                codec_opts = codec_config.get_codec_options(codec_name)
                config.update(codec_opts)
        
        return config
    
    def _codec_packet_to_packet_info(
        self, 
        codec_packet: CodecPacket, 
        stream_index: int, 
        default_timestamp: Optional[int],
        stream: Any = None
    ) -> PacketInfo:
        """Convert a CodecPacket to PacketInfo"""
        # Get time base from stream if available
        if stream is not None and hasattr(stream, 'time_base'):
            time_base = (stream.time_base.numerator, stream.time_base.denominator)
        else:
            time_base = (1, 1000)  # Default millisecond time base
        
        return PacketInfo(
            data=codec_packet.data,
            pts=codec_packet.metadata.get("pts", default_timestamp),
            dts=codec_packet.metadata.get("dts", default_timestamp),
            stream_index=stream_index,
            time_base=time_base,
            is_keyframe=codec_packet.metadata.get("is_keyframe", codec_packet.seekable)
        )
    
    def _packet_info_to_codec_packet(self, packet_info: PacketInfo, codec: DataCodec) -> CodecPacket:
        """Convert PacketInfo to CodecPacket for decoding"""
        return CodecPacket(
            data=packet_info.data,
            metadata={
                "pts": packet_info.pts,
                "dts": packet_info.dts,
                "codec": codec.get_codec_name(),
                "time_base": packet_info.time_base
            },
            seekable=packet_info.is_keyframe
        )
    
    def _fallback_decode(self, packet_info: PacketInfo) -> Any:
        """Fallback decoding using pickle"""
        try:
            import pickle
            return pickle.loads(packet_info.data)
        except Exception as e:
            logger.error(f"Fallback decode failed: {e}")
            return packet_info.data 