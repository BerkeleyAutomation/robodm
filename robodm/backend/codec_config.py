from typing import List, Dict, Any, Optional, Tuple, cast, Union
from fractions import Fraction
import logging
import av
from robodm.feature import FeatureType

logger = logging.getLogger(__name__)


class CodecConfig:
    """Configuration class for video codec settings with feature-specific codec mapping."""

    @staticmethod
    def get_supported_pixel_formats(codec_name: str) -> List[str]:
        """Get list of supported pixel formats for a codec."""
        try:
            import av

            codec = av.codec.Codec(codec_name, "w")
            if codec.video_formats:
                return [vf.name for vf in codec.video_formats]
            return []
        except Exception:
            return []

    @staticmethod
    def is_codec_config_supported(width: int,
                                  height: int,
                                  pix_fmt: str = "yuv420p",
                                  codec_name: str = "libx264") -> bool:
        """Check if a specific width/height/pixel format combination is supported by codec."""
        try:
            cc = av.codec.CodecContext.create(codec_name, "w")
            cc.width = width
            cc.height = height
            cc.pix_fmt = pix_fmt
            cc.time_base = Fraction(1, 30)
            cc.open(strict=True)
            cc.close()
            return True
        except Exception:
            return False

    @staticmethod
    def is_valid_image_shape(shape: Tuple[int, ...],
                             codec_name: str = "libx264") -> bool:
        """Check if a shape can be treated as an RGB image for the given codec."""
        # Only accept RGB shapes (H, W, 3)
        if len(shape) != 3 or shape[2] != 3:
            return False

        height, width = shape[0], shape[1]

        # Check minimum reasonable image size
        if height < 1 or width < 1:
            return False

        # Check codec-specific constraints
        if codec_name in ["libx264", "libx265"]:
            # H.264/H.265 require even dimensions
            if height % 2 != 0 or width % 2 != 0:
                return False
        elif codec_name in ["libaom-av1"]:
            # AV1 also typically requires even dimensions for yuv420p
            if height % 2 != 0 or width % 2 != 0:
                return False

        # Test if the codec actually supports this resolution
        return CodecConfig.is_codec_config_supported(width, height, "yuv420p",
                                                     codec_name)

    # Default codec configurations
    CODEC_CONFIGS = {
        "rawvideo": {
            "pixel_format": None,  # No pixel format for rawvideo (binary)
            "options": {},
            "raw_codec": "pickle_raw",  # Default raw codec implementation
        },
        "rawvideo_pickle": {
            "pixel_format": None,
            "options": {},
            "raw_codec": "pickle_raw",
        },
        "rawvideo_pyarrow": {
            "pixel_format": None,
            "options": {
                "batch_size": 100,
                "compression": "snappy"
            },
            "raw_codec": "pyarrow_batch",
        },
        "libx264": {
            "pixel_format": "yuv420p",
            "options": {
                "crf": "23",
                "preset": "medium"
            },  # Default quality
        },
        "libx265": {
            "pixel_format": "yuv420p",
            "options": {
                "crf": "28",
                "preset": "medium"
            },  # Default quality for HEVC
        },
        "libaom-av1": {
            "pixel_format": "yuv420p",
            "options": {
                "g": "2",
                "crf": "30"
            }
        },
        "ffv1": {
            "pixel_format":
            "yuv420p",  # Default, will be adjusted based on content
            "options": {},
        },
    }

    def __init__(self,
                 codec: Union[str, Dict[str, str]] = "auto",
                 options: Optional[Dict[str, Any]] = None):
        """
        Initialize codec configuration.

        Args:
            codec: Either a default codec string ("auto", "rawvideo", etc.) or 
                   a dictionary mapping feature names to specific codecs {feature_name: codec}
            options: Additional codec-specific options
        """
        if isinstance(codec, dict):
            # Feature-specific codec mapping
            self.feature_codecs = codec
            self.codec = "auto"  # Default for unmapped features
        else:
            # Single codec for all features
            self.codec = codec
            self.feature_codecs = {}
        
        self.custom_options = options or {}

        # Validate all specified codecs
        all_codecs = set([self.codec])
        all_codecs.update(self.feature_codecs.values())
        
        for codec_name in all_codecs:
            if codec_name not in ["auto"] and codec_name not in self.CODEC_CONFIGS:
                raise ValueError(
                    f"Unsupported codec: {codec_name}. Supported: {list(self.CODEC_CONFIGS.keys())}"
                )

    def get_codec_for_feature(self, feature_type: FeatureType, feature_name: Optional[str] = None) -> str:
        """Determine the appropriate codec for a given feature type and name."""
        
        # Check for feature-specific codec mapping first
        if feature_name and feature_name in self.feature_codecs:
            specified_codec = self.feature_codecs[feature_name]
            logger.debug(f"Using feature-specific codec {specified_codec} for {feature_name}")
            
            # Validate the codec can handle this feature type
            if self._can_codec_handle_feature(specified_codec, feature_type):
                return specified_codec
            else:
                logger.warning(
                    f"Feature-specific codec {specified_codec} cannot handle feature {feature_name} "
                    f"with type {feature_type}, falling back to auto-selection"
                )

        # Fall back to default codec selection logic
        data_shape = feature_type.shape

        # Only use video codecs for RGB images (H, W, 3)
        if data_shape is not None and len(
                data_shape) == 3 and data_shape[2] == 3:
            height, width = data_shape[0], data_shape[1]

            # If user specified a codec other than auto, try to use it for RGB images
            if self.codec != "auto":
                # Handle rawvideo variants
                if self.codec.startswith("rawvideo"):
                    return self.codec
                elif self.is_valid_image_shape(data_shape, self.codec):
                    logger.debug(
                        f"Using user-specified codec {self.codec} for RGB shape {data_shape}"
                    )
                    return self.codec
                else:
                    logger.warning(
                        f"User-specified codec {self.codec} doesn't support shape {data_shape}, falling back to rawvideo"
                    )
                    return "rawvideo"

            # Auto-selection for RGB images only
            codec_preferences = ["libaom-av1", "ffv1", "libx264", "libx265"]

            for codec in codec_preferences:
                if self.is_valid_image_shape(data_shape, codec):
                    logger.debug(
                        f"Selected codec {codec} for RGB shape {data_shape}")
                    return codec

            # If no video codec works for this RGB image, fall back to rawvideo
            logger.warning(
                f"No video codec supports RGB shape {data_shape}, falling back to rawvideo"
            )

        else:
            # Non-RGB data (grayscale, depth, vectors, etc.) always use rawvideo
            if data_shape is not None:
                logger.debug(f"Using rawvideo for non-RGB shape {data_shape}")

        return "rawvideo"
    
    def _can_codec_handle_feature(self, codec: str, feature_type: FeatureType) -> bool:
        """Check if a codec can handle a specific feature type."""
        if codec.startswith("rawvideo"):
            # Raw codecs can handle any data type
            return True
        
        # Video codecs can only handle RGB images
        data_shape = feature_type.shape
        if data_shape is not None and len(data_shape) == 3 and data_shape[2] == 3:
            return self.is_valid_image_shape(data_shape, codec)
        
        return False
    
    def get_raw_codec_name(self, codec: str) -> str:
        """Get the raw codec implementation name for a given codec."""
        if codec not in self.CODEC_CONFIGS:
            raise ValueError(f"Unknown codec {codec}")
        
        codec_config = cast(Dict[str, Any], self.CODEC_CONFIGS[codec])
        return codec_config.get("raw_codec", "pickle_raw")

    def get_pixel_format(self, codec: str,
                         feature_type: FeatureType) -> Optional[str]:
        """Get appropriate pixel format for codec and feature type."""
        if codec not in self.CODEC_CONFIGS:
            return None

        codec_config = cast(Dict[str, Any], self.CODEC_CONFIGS[codec])
        base_format = codec_config.get("pixel_format")

        # For FFV1, adjust pixel format based on data type
        if codec == "ffv1" and feature_type.dtype == "uint8":
            data_shape = feature_type.shape
            if data_shape is not None and len(data_shape) == 3:
                if data_shape[2] == 3:  # RGB
                    return "rgb24"
                elif data_shape[2] == 4:  # RGBA
                    return "rgba"

        return base_format

    def get_codec_options(self, codec: str) -> Dict[str, Any]:
        """Get codec options, merging defaults with custom options."""
        if codec not in self.CODEC_CONFIGS:
            return self.custom_options.copy()

        codec_config = cast(Dict[str, Any], self.CODEC_CONFIGS[codec])
        default_options = codec_config.get("options", {}).copy()

        # Merge custom options (custom options override defaults)
        default_options.update(self.custom_options)
        return default_options

    @classmethod
    def from_video_codec(cls, video_codec: str = "auto", codec_options: Optional[Dict[str, Any]] = None) -> "CodecConfig":
        """Create CodecConfig from video_codec parameter (for backward compatibility)."""
        return cls(codec=video_codec, options=codec_options)
