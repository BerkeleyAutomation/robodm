import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_DTYPES = [
    "null",
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "timestamp(s)",
    "timestamp(ms)",
    "timestamp(us)",
    "timestamp(ns)",
    "timestamp(s, tz)",
    "timestamp(ms, tz)",
    "timestamp(us, tz)",
    "timestamp(ns, tz)",
    "binary",
    "large_binary",
    "string",
    "str",
    "large_string",
    "bytes",
]


class FeatureType:
    """
    class for feature definition and conversions
    """

    def __init__(
        self,
        dtype: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        tf_feature_spec=None,
        data=None,
    ) -> None:
        # scalar: (), vector: (n,), matrix: (n,m)
        self.dtype: str = ""
        self.shape: Optional[Tuple[int, ...]] = None

        if data is not None:
            self.from_data(data)
        elif tf_feature_spec is not None:
            self.from_tf_feature_type(tf_feature_spec)
        elif dtype is not None:
            self._set(dtype, shape)

    def __str__(self):
        return f"dtype={self.dtype}; shape={self.shape})"

    def __repr__(self):
        return self.__str__()

    def _set(self, dtype: str, shape: Optional[Tuple[int, ...]]):
        if dtype == "double":  # fix inferred type
            dtype = "float64"
        if dtype == "float":  # fix inferred type
            dtype = "float32"
        if dtype == "int":  # fix inferred type
            dtype = "int32"
        if dtype == "object":
            dtype = "string"
        if dtype == "bytes":
            dtype = "string"
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")
        if shape is not None and not isinstance(shape, tuple):
            raise ValueError(f"Shape must be a tuple: {shape}")
        self.dtype = dtype
        self.shape = shape

    def from_tf_feature_type(self, tf_feature_spec):
        """
        Convert from tf feature
        """
        logger.debug(f"tf_feature_spec: {tf_feature_spec}")
        from tensorflow_datasets.core.features import (FeaturesDict, Image,
                                                       Scalar, Tensor, Text)

        if isinstance(tf_feature_spec, Tensor):
            shape = tf_feature_spec.shape
            dtype = tf_feature_spec.dtype.name
        elif isinstance(tf_feature_spec, Image):
            shape = tf_feature_spec.shape
            dtype = tf_feature_spec.np_dtype
            # TODO: currently images are not handled efficiently
        elif isinstance(tf_feature_spec, Scalar):
            shape = ()
            dtype = tf_feature_spec.dtype.name
        elif isinstance(tf_feature_spec, Text):
            shape = ()
            dtype = "string"
        else:
            raise ValueError(
                f"Unsupported conversion from tf feature: {tf_feature_spec}")
        self._set(str(dtype), shape)
        return self

    @classmethod
    def from_data(cls, data: Any):
        """
        Infer feature type from the provided data.
        """
        feature_type = FeatureType()
        if isinstance(data, np.ndarray):
            feature_type._set(data.dtype.name, data.shape)
        elif isinstance(data, np.bool_):
            feature_type._set("bool", ())
        elif isinstance(data, list):
            dtype = type(data[0]).__name__
            data_shape: Tuple[int, ...] = (len(data), )
            feature_type._set(dtype, data_shape)
        else:
            dtype = type(data).__name__
            if dtype == 'object':
                dtype = 'string'
            empty_shape: Tuple[int, ...] = ()
            try:
                feature_type._set(dtype, empty_shape)
            except ValueError as e:
                print(f"Error: {e}")
                print(f"dtype: {dtype}")
                print(f"shape: {empty_shape}")
                print(f"data: {data}")
                raise e
        return feature_type

    @classmethod
    def from_str(cls, feature_str: str):
        """
        Parse a string representation of the feature type.
        """
        dtype, shape_str = feature_str.split(";")
        dtype = dtype.split("=")[1]
        shape_eval = eval(shape_str.split("=")[1][:-1])  # strip brackets
        # Ensure shape is a tuple
        if isinstance(shape_eval, tuple):
            shape: Optional[Tuple[int, ...]] = shape_eval
        else:
            shape = None
        return FeatureType(dtype=dtype, shape=shape)

    def to_tf_feature_type(self, first_dim_none=False):
        """
        Convert to tf feature
        """
        import tensorflow as tf
        from tensorflow_datasets.core.features import (FeaturesDict, Image,
                                                       Scalar, Tensor, Text)

        str_dtype_to_tf_dtype = {
            "int8": tf.int8,
            "int16": tf.int16,
            "int32": tf.int32,
            "int64": tf.int64,
            "uint8": tf.uint8,
            "uint16": tf.uint16,
            "uint32": tf.uint32,
            "uint64": tf.uint64,
            "float16": tf.float16,
            "float32": tf.float32,
            "float64": tf.float64,
            "string": tf.string,
            "str": tf.string,
            "bool": tf.bool,
        }
        tf_detype = str_dtype_to_tf_dtype[self.dtype]
        if self.shape is not None and len(self.shape) == 0:
            if self.dtype == "string":
                return Text()
            else:
                return Scalar(dtype=tf_detype)
        elif self.shape is not None and len(self.shape) >= 1:
            if first_dim_none:
                tf_shape = [None] + list(self.shape[1:])
                return Tensor(shape=tf_shape, dtype=tf_detype)
            else:
                return Tensor(shape=self.shape, dtype=tf_detype)
        else:
            raise ValueError(f"Unsupported conversion to tf feature: {self}")

    def to_pld_storage_type(self):
        if self.shape is not None and len(self.shape) == 0:
            if self.dtype == "string":
                return "large_binary"  # TODO: better representation of strings
            else:
                return self.dtype
        else:
            return "large_binary"
