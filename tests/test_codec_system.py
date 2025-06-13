"""
Test cases for the codec abstraction system.

This module tests the extensible codec system including:
- Codec registration and factory
- Codec manager functionality
- Individual codec implementations
- Integration with backend
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

# Import the codec system components
from robodm.backend.codec_interface import DataCodec, RawDataCodec, VideoCodec, CodecPacket
from robodm.backend.codecs import (
    register_codec, get_codec, list_available_codecs, clear_codec_cache,
    is_video_codec, is_raw_codec, PickleRawCodec, PyAVVideoCodec
)
from robodm.backend.codec_manager import CodecManager
from robodm.backend.base import PacketInfo
from robodm.backend.codec_config import CodecConfig


class MockRawCodec(RawDataCodec):
    """Mock raw codec for testing"""
    
    def __init__(self, name: str = "mock_raw", **kwargs):
        self.name = name
        self.options = kwargs
        self.encoded_data = []
        self.flushed = False
    
    def encode(self, data, timestamp, **kwargs):
        packet = CodecPacket(
            data=f"encoded_{data}_{timestamp}".encode(),
            metadata={"pts": timestamp, "dts": timestamp, "codec": self.name},
            seekable=True
        )
        self.encoded_data.append((data, timestamp))
        return [packet]
    
    def decode(self, packet):
        # Simple mock decoding
        data_str = packet.data.decode()
        if data_str.startswith("encoded_"):
            parts = data_str.split("_")
            return f"decoded_{parts[1]}"
        return packet.data.decode()
    
    def flush(self):
        self.flushed = True
        return []
    
    def supports_seeking(self):
        return True
    
    def get_codec_name(self):
        return self.name
    
    def get_container_encoding(self):
        return "rawvideo"


class MockVideoCodec(VideoCodec):
    """Mock video codec for testing"""
    
    def __init__(self, codec_name: str = "mock_video", **kwargs):
        self.codec_name = codec_name
        self.config = kwargs
        self.stream = None
        self.encoded_frames = []
    
    def configure_stream(self, stream, feature_type):
        self.stream = stream
        
    def create_frame(self, data, timestamp):
        return Mock(pts=timestamp, data=data)
    
    def encode(self, data, timestamp, **kwargs):
        packet = CodecPacket(
            data=f"video_encoded_{self.codec_name}_{timestamp}".encode(),
            metadata={"pts": timestamp, "dts": timestamp, "codec": self.codec_name, "is_keyframe": True},
            seekable=True
        )
        self.encoded_frames.append((data, timestamp))
        return [packet]
    
    def decode(self, packet):
        return f"video_decoded_{packet.data.decode()}"
    
    def flush(self):
        return []
    
    def supports_seeking(self):
        return True
    
    def get_codec_name(self):
        return self.codec_name


class TestCodecRegistry:
    """Test codec registration and factory functionality"""
    
    def setup_method(self):
        """Clear codec cache before each test"""
        clear_codec_cache()
    
    def test_register_codec(self):
        """Test codec registration"""
        register_codec("test_mock", MockRawCodec)
        
        # Check that codec is registered
        assert "test_mock" in list_available_codecs()
        
        # Create instance
        codec = get_codec("test_mock", name="test_instance")
        assert isinstance(codec, MockRawCodec)
        assert codec.name == "test_instance"
    
    def test_register_invalid_codec(self):
        """Test that registering invalid codec raises error"""
        with pytest.raises(TypeError):
            register_codec("invalid", str)  # Not a DataCodec subclass
    
    def test_get_unknown_codec(self):
        """Test getting unknown codec raises error"""
        with pytest.raises(ValueError, match="Unknown codec: nonexistent"):
            get_codec("nonexistent")
    
    def test_codec_caching(self):
        """Test that codec instances are cached"""
        register_codec("cached_test", MockRawCodec)
        
        codec1 = get_codec("cached_test", name="test")
        codec2 = get_codec("cached_test", name="test")
        
        # Should be the same instance
        assert codec1 is codec2
    
    def test_codec_type_checking(self):
        """Test codec type checking functions"""
        register_codec("raw_test", MockRawCodec)
        register_codec("video_test", MockVideoCodec)
        
        assert is_raw_codec("raw_test")
        assert not is_video_codec("raw_test")
        
        assert is_video_codec("video_test")
        assert not is_raw_codec("video_test")
        
        assert not is_raw_codec("nonexistent")
        assert not is_video_codec("nonexistent")


class TestPickleRawCodec:
    """Test the pickle raw codec implementation"""
    
    def test_encode_decode_numpy(self):
        """Test encoding/decoding numpy arrays"""
        codec = PickleRawCodec()
        data = np.array([1, 2, 3, 4, 5])
        timestamp = 1000
        
        # Encode
        packets = codec.encode(data, timestamp)
        assert len(packets) == 1
        
        packet = packets[0]
        assert packet.metadata["pts"] == timestamp
        assert packet.metadata["codec"] == "pickle_raw"
        assert not packet.seekable
        
        # Decode
        decoded = codec.decode(packet)
        np.testing.assert_array_equal(decoded, data)
    
    def test_encode_decode_complex_object(self):
        """Test encoding/decoding complex Python objects"""
        codec = PickleRawCodec()
        data = {"key": [1, 2, 3], "nested": {"value": 42}}
        timestamp = 2000
        
        # Encode
        packets = codec.encode(data, timestamp)
        assert len(packets) == 1
        
        # Decode
        decoded = codec.decode(packets[0])
        assert decoded == data
    
    def test_flush(self):
        """Test flushing (should return empty list)"""
        codec = PickleRawCodec()
        assert codec.flush() == []
    
    def test_properties(self):
        """Test codec properties"""
        codec = PickleRawCodec()
        assert codec.get_codec_name() == "pickle_raw"
        assert codec.get_container_encoding() == "rawvideo"
        assert not codec.supports_seeking()


@pytest.mark.skipif(
    not hasattr(pytest, "importorskip") or 
    pytest.importorskip("pyarrow", reason="PyArrow not available"),
    reason="PyArrow not available"
)
class TestPyArrowBatchCodec:
    """Test the PyArrow batch codec implementation"""
    
    def test_batch_encoding(self):
        """Test batching behavior"""
        from robodm.backend.codecs import PyArrowBatchCodec
        
        codec = PyArrowBatchCodec(batch_size=3)
        
        # Add data points - should not produce packets until batch is full
        packets1 = codec.encode(np.array([1, 2]), 1000)
        assert len(packets1) == 0
        
        packets2 = codec.encode(np.array([3, 4]), 2000)
        assert len(packets2) == 0
        
        # Third item should trigger batch flush
        packets3 = codec.encode(np.array([5, 6]), 3000)
        assert len(packets3) == 1
        
        # Check packet metadata
        packet = packets3[0]
        assert packet.metadata["batch_size"] == 3
        assert packet.metadata["batch_start_pts"] == 1000
        assert packet.metadata["batch_end_pts"] == 3000
        assert packet.seekable
    
    def test_decode_batch(self):
        """Test decoding batched data"""
        from robodm.backend.codecs import PyArrowBatchCodec
        
        codec = PyArrowBatchCodec(batch_size=2)
        
        # Encode some data
        codec.encode(np.array([1, 2]), 1000)
        packets = codec.encode(np.array([3, 4]), 2000)
        
        # Decode the batch
        decoded_items = codec.decode(packets[0])
        assert len(decoded_items) == 2
        
        pts1, data1 = decoded_items[0]
        pts2, data2 = decoded_items[1]
        
        assert pts1 == 1000
        np.testing.assert_array_equal(data1, np.array([1, 2]))
        
        assert pts2 == 2000
        np.testing.assert_array_equal(data2, np.array([3, 4]))
    
    def test_flush_partial_batch(self):
        """Test flushing incomplete batch"""
        from robodm.backend.codecs import PyArrowBatchCodec
        
        codec = PyArrowBatchCodec(batch_size=5)
        
        # Add some data (less than batch size)
        codec.encode(np.array([1, 2]), 1000)
        codec.encode(np.array([3, 4]), 2000)
        
        # Flush should return the partial batch
        packets = codec.flush()
        assert len(packets) == 1
        
        # Decode and verify
        decoded_items = codec.decode(packets[0])
        assert len(decoded_items) == 2


class TestCodecManager:
    """Test the codec manager functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        clear_codec_cache()
        register_codec("test_raw", MockRawCodec)
        register_codec("test_video", MockVideoCodec)
        # Register video codecs with their actual names for testing
        register_codec("libx264", MockVideoCodec)
        register_codec("libx265", MockVideoCodec)
        register_codec("libaom-av1", MockVideoCodec)
        register_codec("ffv1", MockVideoCodec)
        self.manager = CodecManager()
        self.mock_config = Mock()
        self.mock_config.get_raw_codec_name.return_value = "test_raw"
        self.mock_config.get_codec_options.return_value = {}
    
    def test_create_raw_codec_for_stream(self):
        """Test creating raw codec for stream"""
        stream_index = 0
        encoding = "rawvideo"
        
        codec = self.manager.create_codec_for_stream(
            stream_index, encoding, self.mock_config
        )
        
        assert codec is not None
        assert isinstance(codec, MockRawCodec)
        assert self.manager.get_codec_for_stream(stream_index) is codec
    
    def test_create_video_codec_for_stream(self):
        """Test creating video codec for stream"""
        stream_index = 1
        encoding = "libx264"
        mock_stream = Mock()
        
        # Mock the config methods for video codec
        self.mock_config.get_pixel_format.return_value = "yuv420p"
        self.mock_config.get_codec_options.return_value = {"crf": "23"}
        
        codec = self.manager.create_codec_for_stream(
            stream_index, encoding, self.mock_config, stream=mock_stream
        )
        
        assert codec is not None
        assert isinstance(codec, MockVideoCodec)
        assert codec.codec_name == "libx264"
    
    def test_encode_data(self):
        """Test encoding data through manager"""
        stream_index = 0
        encoding = "rawvideo"
        
        # Create codec
        self.manager.create_codec_for_stream(stream_index, encoding, self.mock_config)
        
        # Mock stream for time base
        mock_stream = Mock()
        mock_stream.time_base.numerator = 1
        mock_stream.time_base.denominator = 1000
        
        # Encode data
        data = "test_data"
        timestamp = 5000
        packets = self.manager.encode_data(stream_index, data, timestamp, mock_stream)
        
        assert len(packets) == 1
        packet = packets[0]
        assert isinstance(packet, PacketInfo)
        assert packet.pts == timestamp
        assert packet.stream_index == stream_index
    
    def test_flush_stream(self):
        """Test flushing stream through manager"""
        stream_index = 0
        encoding = "rawvideo"
        
        # Create codec
        codec = self.manager.create_codec_for_stream(stream_index, encoding, self.mock_config)
        
        # Flush
        packets = self.manager.flush_stream(stream_index)
        assert isinstance(packets, list)
        assert codec.flushed
    
    def test_decode_packet(self):
        """Test decoding packet through manager"""
        stream_index = 0
        encoding = "rawvideo"
        
        # Create codec
        self.manager.create_codec_for_stream(stream_index, encoding, self.mock_config)
        
        # Create a PacketInfo to decode
        packet_info = PacketInfo(
            data=b"encoded_test_data_1000",
            pts=1000,
            dts=1000,
            stream_index=stream_index,
            time_base=(1, 1000),
            is_keyframe=True
        )
        
        # Decode
        result = self.manager.decode_packet(packet_info)
        assert result == "decoded_test"  # Based on MockRawCodec logic
    
    def test_get_codec_info(self):
        """Test getting codec information"""
        stream_index = 0
        encoding = "rawvideo"
        
        # Create codec
        self.manager.create_codec_for_stream(stream_index, encoding, self.mock_config)
        
        # Get info
        info = self.manager.get_codec_info(stream_index)
        assert info is not None
        assert info["codec_name"] == "mock_raw"  # MockRawCodec returns "mock_raw" by default
        assert info["supports_seeking"] is True
        assert info["is_raw_codec"] is True
        assert info["is_video_codec"] is False
    
    def test_clear_stream_codecs(self):
        """Test clearing all stream codecs"""
        # Create some codecs
        self.manager.create_codec_for_stream(0, "rawvideo", self.mock_config)
        self.manager.create_codec_for_stream(1, "rawvideo", self.mock_config)
        
        assert self.manager.get_codec_for_stream(0) is not None
        assert self.manager.get_codec_for_stream(1) is not None
        
        # Clear
        self.manager.clear_stream_codecs()
        
        assert self.manager.get_codec_for_stream(0) is None
        assert self.manager.get_codec_for_stream(1) is None


class TestCodecIntegration:
    """Integration tests for codec system with backend"""
    
    def setup_method(self):
        """Setup for integration tests"""
        clear_codec_cache()
    
    @patch('robodm.backend.pyav_backend.av')
    def test_backend_codec_integration(self, mock_av):
        """Test integration between backend and codec system"""
        from robodm.backend.pyav_backend import PyAVBackend
        from robodm.backend.codec_config import CodecConfig
        
        # Mock PyAV objects
        mock_container = Mock()
        mock_stream = Mock()
        mock_stream.index = 0
        mock_stream.codec_context.codec.name = "rawvideo"
        mock_stream.metadata = {"FEATURE_NAME": "test", "ORIGINAL_CODEC": "rawvideo"}
        mock_stream.time_base.numerator = 1
        mock_stream.time_base.denominator = 1000
        
        mock_container.streams = [mock_stream]
        mock_av.open.return_value = mock_container
        
        # Create backend
        backend = PyAVBackend()
        backend.open("test.vla", "w")
        backend._idx_to_stream[0] = mock_stream
        
        # Create codec config
        codec_config = CodecConfig(codec="rawvideo")
        
        # Test encoding
        data = np.array([1, 2, 3])
        timestamp = 1000
        packets = backend.encode_data_to_packets(data, 0, timestamp, codec_config)
        
        # Should fall back to legacy behavior when codec creation fails
        assert len(packets) >= 1
        
        backend.close()
    
    def test_codec_config_integration(self):
        """Test integration with codec configuration"""
        from robodm.backend.codec_config import CodecConfig
        
        # Test rawvideo codec selection
        config = CodecConfig(codec="rawvideo_pickle")
        assert config.get_raw_codec_name("rawvideo_pickle") == "pickle_raw"
        
        # Test with PyArrow
        try:
            import pyarrow
            config_arrow = CodecConfig(codec="rawvideo_pyarrow")
            assert config_arrow.get_raw_codec_name("rawvideo_pyarrow") == "pyarrow_batch"
        except ImportError:
            pass  # Skip if PyArrow not available


class TestExtensibility:
    """Test the extensibility of the codec system"""
    
    def setup_method(self):
        clear_codec_cache()
    
    def test_custom_codec_registration(self):
        """Test that custom codecs can be easily registered and used"""
        
        class CustomCodec(RawDataCodec):
            def __init__(self, prefix="custom", **kwargs):
                self.prefix = prefix
                
            def encode(self, data, timestamp, **kwargs):
                encoded_data = f"{self.prefix}:{data}:{timestamp}".encode()
                return [CodecPacket(
                    data=encoded_data,
                    metadata={"pts": timestamp, "dts": timestamp},
                    seekable=True
                )]
            
            def decode(self, packet):
                parts = packet.data.decode().split(":")
                return parts[1]  # Return original data part
            
            def flush(self):
                return []
            
            def supports_seeking(self):
                return True
            
            def get_codec_name(self):
                return f"custom_{self.prefix}"
            
            def get_container_encoding(self):
                return "rawvideo"
        
        # Register custom codec
        register_codec("my_custom", CustomCodec)
        
        # Use it
        codec = get_codec("my_custom", prefix="test")
        assert codec.prefix == "test"
        
        # Test encoding/decoding
        packets = codec.encode("hello", 1000)
        assert len(packets) == 1
        
        decoded = codec.decode(packets[0])
        assert decoded == "hello"
    
    def test_codec_manager_with_custom_codec(self):
        """Test codec manager works with custom codecs"""
        
        class SimpleCodec(RawDataCodec):
            def __init__(self, multiplier=1, **kwargs):
                self.multiplier = multiplier
            
            def encode(self, data, timestamp, **kwargs):
                # Simple transformation
                transformed = data * self.multiplier if hasattr(data, '__mul__') else data
                return [CodecPacket(
                    data=str(transformed).encode(),
                    metadata={"pts": timestamp},
                    seekable=False
                )]
            
            def decode(self, packet):
                return packet.data.decode()
            
            def flush(self):
                return []
            
            def supports_seeking(self):
                return False
            
            def get_codec_name(self):
                return "simple"
            
            def get_container_encoding(self):
                return "rawvideo"
        
        # Register and test
        register_codec("simple", SimpleCodec)
        
        manager = CodecManager()
        mock_config = Mock()
        mock_config.get_raw_codec_name.return_value = "simple"
        mock_config.get_codec_options.return_value = {"multiplier": 3}
        
        # Create codec through manager
        codec = manager.create_codec_for_stream(0, "rawvideo", mock_config)
        assert codec is not None
        assert codec.multiplier == 3
        
        # Test encoding through manager
        packets = manager.encode_data(0, 5, 1000)
        assert len(packets) == 1
        
        # The encoded data should be "15" (5 * 3)
        decoded = manager.decode_packet(packets[0])
        assert decoded == "15"


if __name__ == "__main__":
    pytest.main([__file__]) 