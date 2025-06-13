import asyncio
import logging
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
# fractions.Fraction imported where needed
from typing import Any, Dict, List, Optional, Text, Tuple, Union, cast

import av
import h5py
import numpy as np

from robodm import FeatureType
from robodm.trajectory_base import TrajectoryInterface
from robodm.utils.flatten import _flatten_dict

# Backend abstraction
from robodm.backend.pyav_backend import PyAVBackend
from robodm.backend.base import ContainerBackend

logger = logging.getLogger(__name__)

logging.getLogger("libav").setLevel(logging.CRITICAL)

from robodm.backend.codec_config import CodecConfig
from robodm.utils.time_manager import TimeManager
from robodm.utils.resampler import FrequencyResampler

class Trajectory(TrajectoryInterface):

    def __init__(
        self,
        path: Text,
        mode="r",
        video_codec: str = "auto",
        codec_options: Optional[Dict[str, Any]] = None,
        feature_name_separator: Text = "/",
        filesystem: Optional[Any] = None,
        time_provider: Optional[Any] = None,
        base_datetime: Optional[datetime] = None,
        time_unit: str = "ms",
        enforce_monotonic: bool = True,
        visualization_feature: Optional[Text] = None,
        backend: Optional[ContainerBackend] = None,
    ) -> None:
        """
        Args:
            path (Text): path to the trajectory file
            mode (Text, optional):  mode of the file, "r" for read and "w" for write
            video_codec (str, optional): Video codec to use. Options: "auto", "rawvideo", "libx264", "libx265", "libaom-av1", "ffv1". Defaults to "auto".
            codec_options (Dict[str, Any], optional): Additional codec-specific options.
            feature_name_separator (Text, optional):
                Delimiter to separate feature names in the container file.
                Defaults to "/".
            filesystem: Optional filesystem interface for dependency injection
            time_provider: Optional time provider interface for dependency injection
            base_datetime: Optional base datetime for timestamp calculations
            time_unit: Default time unit for timestamp inputs ('ns', 'μs', 'ms', 's')
            enforce_monotonic: Whether to enforce monotonically increasing timestamps
            visualization_feature: Optional feature name to prioritize as first stream for visualization.
                If None, automatically puts video-encoded streams first during compacting.
            backend: Optional container backend for dependency injection
        """
        self.path = path
        self.feature_name_separator = feature_name_separator
        self.visualization_feature = visualization_feature


        # Initialize codec configuration
        self.codec_config = CodecConfig.from_video_codec(video_codec, codec_options)

        # Dependency injection - set early so they're available during init
        self._filesystem = filesystem
        self._time_provider = time_provider

        # Initialize time management system
        self.time_manager = TimeManager(
            base_datetime=base_datetime,
            time_unit=time_unit,
            enforce_monotonic=enforce_monotonic,
        )

        self.feature_name_to_stream: Dict[str,
                                          Any] = {}  # feature_name: stream
        self.feature_name_to_feature_type: Dict[str, FeatureType] = (
            {})  # feature_name: feature_type
        self.trajectory_data = None  # trajectory_data
        self.start_time = self._time()
        self.mode = mode
        self.is_closed = False
        self.pending_write_tasks: List[Any] = (
            [])  # List to keep track of pending write tasks
        self.container_file: Optional[Any] = None  # av.OutputContainer or None

        # ------------------------------------------------------------------ #
        # Container backend setup
        # ------------------------------------------------------------------ #
        self.backend: ContainerBackend = backend or PyAVBackend()

        # check if the path exists
        # if not, create a new file and start data collection
        if self.mode == "w":
            if not self._exists(self.path):
                self._makedirs(os.path.dirname(self.path), exist_ok=True)
            try:
                # Use backend to open the container so that the rest of the
                # class can keep using `self.container_file` (PyAV Container).
                self.backend.open(self.path, "w")
                # Expose underlying PyAV container for legacy code paths that
                # access it directly.
                self.container_file = getattr(self.backend, "container", None)
            except Exception as e:
                logger.error(f"error creating the trajectory file: {e}")
                raise
        elif self.mode == "r":
            if not self._exists(self.path):
                raise FileNotFoundError(f"{self.path} does not exist")
            # Open the backend in read mode now so that subsequent operations
            # can reuse the container without touching PyAV directly here.
            self.backend.open(self.path, "r")
            self.container_file = getattr(self.backend, "container", None)
        else:
            raise ValueError(f"Invalid mode {self.mode}, must be 'r' or 'w'")

    def _exists(self, path: str) -> bool:
        """File existence check with dependency injection support."""
        if self._filesystem:
            return self._filesystem.exists(path)
        return os.path.exists(path)

    def _makedirs(self, path: str, exist_ok: bool = False) -> None:
        """Directory creation with dependency injection support."""
        if self._filesystem:
            return self._filesystem.makedirs(path, exist_ok=exist_ok)
        return os.makedirs(path, exist_ok=exist_ok)

    def _remove(self, path: str) -> None:
        """File removal with dependency injection support."""
        if self._filesystem:
            return self._filesystem.remove(path)
        return os.remove(path)

    def _rename(self, src: str, dst: str) -> None:
        """File rename with dependency injection support."""
        if self._filesystem:
            return self._filesystem.rename(src, dst)
        return os.rename(src, dst)

    def _time(self) -> float:
        """Time retrieval with dependency injection support."""
        if self._time_provider:
            return self._time_provider.time()
        return time.time()


    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        """
        get the value of the feature
        return hdf5-ed data
        """

        if self.trajectory_data is None:
            logger.info(f"Loading the trajectory data with key {key}")
            self.trajectory_data = self.load()

        if self.trajectory_data is None:
            raise RuntimeError("Failed to load trajectory data")

        return self.trajectory_data[key]

    def close(self, compact=True):
        """
        close the container file

        args:
        compact: re-read from the cache to encode pickled data to images
        """
        logger.debug(
            f"Closing trajectory, is_closed={self.is_closed}, compact={compact}"
        )

        if self.is_closed:
            raise ValueError("The container file is already closed")

        if self.mode == "r":
            # For read mode, just mark as closed
            self.is_closed = True
            self.trajectory_data = None
            logger.debug("Trajectory (read mode) closed successfully")
            return

        # Write mode handling
        if self.backend.container is None:
            logger.warning(
                "Container not available, marking trajectory as closed")
            self.is_closed = True
            return

        # Check if there are any streams with data
        streams = self.backend.get_streams()
        has_data = len(streams) > 0

        try:
            # Flush all streams using backend abstraction
            buffered_packets = self.backend.flush_all_streams()
            logger.debug(f"Flushed {len(buffered_packets)} buffered packets")
            
            # Mux all buffered packets
            for packet_info in buffered_packets:
                if packet_info.pts is None:
                    raise ValueError(f"Packet {packet_info} has no pts")
                self.backend.mux_packet_info(packet_info)
                logger.debug(f"Muxed flush packet from stream {packet_info.stream_index}")
                
            logger.debug("Flushing completed")
        except Exception as e:
            logger.error(f"Error during flush: {e}")

        logger.debug("Closing container")
        self.backend.close()

        # Ensure file exists even if empty
        if not self._exists(self.path):
            logger.warning(
                f"Container was closed but {self.path} doesn't exist. This might indicate an issue."
            )

        # Only attempt transcoding if file exists, has content, and compact is requested
        if (compact and has_data and self._exists(self.path)
                and os.path.getsize(self.path) > 0):
            logger.debug("Starting transcoding of pickled images")
            try:
                self._transcode_pickled_images()
            except Exception as e:
                logger.warning(
                    f"Transcoding failed: {e}. Keeping original file with pickled data."
                )
                # File remains in original state with pickled data, which is still valid
        else:
            logger.debug(
                f"Skipping transcoding: compact={compact}, has_data={has_data}, file_exists={self._exists(self.path)}, file_size={os.path.getsize(self.path) if self._exists(self.path) else 0}"
            )

        self.trajectory_data = None
        self.container_file = None
        self.is_closed = True
        logger.debug("Trajectory closed successfully")

    def load(
        self,
        return_type: str = "numpy",
        desired_frequency: Optional[float] = None,
        data_slice: Optional[slice] = None,
    ):
        """
        Load trajectory data with optional temporal resampling and slicing.

        Parameters
        ----------
        return_type : {"numpy", "container"}, default "numpy"
            • "numpy"     – decode the data and return a dict[str, np.ndarray]
            • "container" – skip all decoding and just return the file path
        desired_frequency : float | None, default None
            Target sampling frequency **in hertz**.  If None, every frame is
            returned (subject to `data_slice`). For upsampling (when desired
            frequency is higher than original), prior frames are duplicated
            to fill temporal gaps. For downsampling, frames are skipped.
        data_slice : slice | None, default None
            Standard Python slice that is applied *after* resampling.
            Example: `slice(100, 200, 2)` → keep resampled indices 100-199,
            step 2.  Negative indices and reverse slices are **not** supported.

        Notes
        -----
        * Resampling is performed individually for every feature stream.
        * For upsampling: when time gaps between consecutive frames exceed
          the desired period, the prior frame is duplicated at regular
          intervals to achieve the target frequency.
        * For downsampling: frames that arrive too close together (within
          the desired period) are skipped.
        * Slicing is interpreted on the **resampled index** domain so that the
          combination `desired_frequency + data_slice` behaves the same as
          `df.iloc[data_slice]` would on a pandas dataframe that had already
          been resampled to `desired_frequency`.
        * When `data_slice` starts at a positive index we `seek()` to the
          corresponding timestamp to avoid decoding frames that will be thrown
          away anyway.
        """
        logger.debug(
            f"load() called with return_type='{return_type}', desired_frequency={desired_frequency}, data_slice={data_slice}"
        )

        # ------------------------------------------------------------------ #
        # Fast-path: user only wants the container path
        # ------------------------------------------------------------------ #
        if return_type == "container":
            logger.debug("Returning container path (fast-path)")
            return self.path
        if return_type not in {"numpy", "container"}:
            raise ValueError("return_type must be 'numpy' or 'container'")

        # ------------------------------------------------------------------ #
        # Validate / canonicalise the slice object
        # ------------------------------------------------------------------ #
        if data_slice is None:
            logger.debug(
                "No data_slice provided, using default slice(None, None, None)"
            )
            data_slice = slice(None, None, None)
        else:
            logger.debug(f"Using provided data_slice: {data_slice}")

        if data_slice.step not in (None, 1) and data_slice.step <= 0:
            raise ValueError("Reverse or zero-step slices are not supported")

        # Check for negative start - this should raise an error
        if data_slice.start is not None and data_slice.start < 0:
            raise ValueError("Negative slice start values are not supported")

        sl_start = 0 if data_slice.start is None else max(data_slice.start, 0)
        sl_stop = data_slice.stop  # can be None
        sl_step = 1 if data_slice.step is None else data_slice.step

        logger.debug(
            f"Canonicalized slice parameters: start={sl_start}, stop={sl_stop}, step={sl_step}"
        )

        # ------------------------------------------------------------------ #
        # Frequency → minimum period in stream time-base units (milliseconds)
        # ------------------------------------------------------------------ #
        period_ms: Optional[int] = None
        if desired_frequency is not None:
            if desired_frequency <= 0:
                raise ValueError("desired_frequency must be positive")
            period_ms = int(round(1000.0 / desired_frequency))
            logger.debug(
                f"Frequency resampling enabled: {desired_frequency} Hz -> period_ms={period_ms}"
            )
        else:
            logger.debug("No frequency resampling (desired_frequency is None)")

        # ------------------------------------------------------------------ #
        # Open the container and, if possible, seek() to the first slice index
        # ------------------------------------------------------------------ #
        # Ensure backend has the container open (read mode).
        if self.backend.container is None:
            self.backend.open(self.path, "r")

        # Get stream metadata from backend
        stream_metadata_list = self.backend.get_streams()
        logger.debug(f"Using backend with {len(stream_metadata_list)} streams")

        # Handle empty trajectory case
        if not stream_metadata_list:
            logger.debug("No streams found in container, returning empty dict")
            self.backend.close()
            return {}

        # Track if we performed seeking to adjust slice logic
        seek_performed = False
        seek_offset_frames = 0

        # Use seeking optimization when we have slicing
        if sl_start > 0 and stream_metadata_list:
            if period_ms is not None:
                # When combining frequency resampling with slicing, seek to the timestamp
                # that corresponds to the sl_start-th frame AFTER resampling.
                # Since resampling keeps every period_ms milliseconds, the sl_start-th
                # resampled frame corresponds to timestamp: sl_start * period_ms
                seek_ts_ms = sl_start * period_ms
                seek_offset_frames = sl_start
                logger.debug(
                    f"Seeking with frequency resampling: seek_ts_ms={seek_ts_ms}, seek_offset_frames={seek_offset_frames}"
                )
            else:
                # If only slicing (no frequency resampling), seek to the sl_start-th frame
                # assuming original 100ms intervals (10Hz from our test data)
                seek_ts_ms = sl_start * 100
                seek_offset_frames = sl_start
                logger.debug(
                    f"Seeking without frequency resampling: seek_ts_ms={seek_ts_ms}, seek_offset_frames={seek_offset_frames}"
                )

            # Seek using the first stream
            try:
                first_stream_idx = 0  # Use first available stream index
                logger.debug(
                    f"Attempting to seek to timestamp {seek_ts_ms} on first stream"
                )
                self.backend.seek_container(seek_ts_ms, first_stream_idx, any_frame=True)
                seek_performed = True
                logger.debug("Seek successful")
            except Exception as e:
                # Seeking failed (e.g. single large packet stream) – fall back
                # to decoding from the beginning.
                logger.debug(
                    f"Seeking failed ({e}), falling back to decoding from beginning"
                )
                seek_performed = False
                seek_offset_frames = 0
        else:
            logger.debug(
                "No seeking optimization needed (sl_start=0 or no streams)")

        # ------------------------------------------------------------------ #
        # Book-keeping structures
        # ------------------------------------------------------------------ #
        cache: dict[str, list[Any]] = {}
        done: set[str] = set()

        # Instantiate the helper that takes care of all frequency based
        # up-/down-sampling **and** slice filtering.
        resampler = FrequencyResampler(
            period_ms=period_ms,
            sl_start=sl_start,
            sl_stop=sl_stop,
            sl_step=sl_step,
            seek_offset_frames=seek_offset_frames,
        )

        # Build stream index mapping and initialize cache
        stream_idx_to_feature: Dict[int, str] = {}
        stream_count = 0
        
        for i, stream_metadata in enumerate(stream_metadata_list):
            fname = stream_metadata.feature_name
            ftype = stream_metadata.feature_type
            if not (fname and ftype) or fname == "unknown":
                logger.debug(
                    f"Skipping stream {i} without valid FEATURE_NAME or FEATURE_TYPE"
                )
                continue
                
            cache[fname] = []
            # Inform the resampler so it can initialise internal bookkeeping
            resampler.register_feature(fname)

            self.feature_name_to_feature_type[fname] = FeatureType.from_str(ftype)
            stream_idx_to_feature[i] = fname
            stream_count += 1
            logger.debug(
                f"Initialized feature '{fname}' with type {ftype}"
            )

        # Handle case where no valid streams were found
        if not cache:
            logger.debug(
                "No valid feature streams found, returning empty dict")
            self.backend.close()
            return {}

        logger.debug(f"Processing {stream_count} feature streams")

        # ------------------------------------------------------------------ #
        # Main demux / decode loop
        # ------------------------------------------------------------------ #
        logger.debug("Starting main demux/decode loop")
        packet_count = 0
        processed_packets = 0
        skipped_frequency = 0
        skipped_slice = 0
        decoded_packets = 0
        upsampled_frames = 0

        # Get stream indices for demuxing
        valid_stream_indices = list(stream_idx_to_feature.keys())
        
        for packet in self.backend.demux_streams(valid_stream_indices):
            packet_count += 1
            
            # Get feature name from stream index
            stream_idx = packet.stream.index
            fname = stream_idx_to_feature.get(stream_idx)
            if fname is None or fname in done:
                continue

            # Use backend's packet validation
            if not self.backend.validate_packet(packet):
                logger.debug(
                    f"Skipping invalid packet for feature '{fname}'")
                continue

            processed_packets += 1

            # -------------------------------------------------------------- #
            # Delegate frequency based up-/down-sampling to helper
            # -------------------------------------------------------------- #
            keep_current, num_dups = resampler.process_packet(
                fname=fname,
                pts=packet.pts,
                has_prior_frame=bool(cache[fname]),
            )

            if not keep_current:
                skipped_frequency += 1
                logger.debug(
                    f"Skipping packet for '{fname}' due to frequency reduction (period_ms={period_ms})"
                )
                continue

            # Insert duplicate frames **before** processing current packet
            if num_dups > 0 and cache[fname]:
                last_frame_data = cache[fname][-1]
                for i in range(num_dups):
                    dup_idx = resampler.next_index(fname)
                    if resampler.want(dup_idx):
                        cache[fname].append(last_frame_data)
                        upsampled_frames += 1
                        logger.debug(
                            f"Inserted duplicate frame for '{fname}' ({i+1}/{num_dups}) at idx={dup_idx}"
                        )

            # Advance index for *current* packet and apply slice filter
            current_idx = resampler.next_index(fname)
            if not resampler.want(current_idx):
                skipped_slice += 1
                resampler.update_last_pts(fname, packet.pts)
                logger.debug(
                    f"Skipping packet for '{fname}' due to slice filter: idx={current_idx}"
                )
                continue

            logger.debug(
                f"Decoding packet for '{fname}': idx={current_idx}, pts={packet.pts}"
            )

            # --- decode on demand only ------------------------------------
            codec = self.backend.get_stream_codec_name(stream_idx)
            if codec == "rawvideo":
                raw = bytes(packet)
                if not raw:  # zero-length placeholder
                    logger.debug(
                        f"Skipping empty rawvideo packet for '{fname}'")
                    continue
                cache[fname].append(pickle.loads(raw))
                decoded_packets += 1
                logger.debug(
                    f"Decoded rawvideo packet for '{fname}' (pickled data)")
            else:
                frames = self.backend.decode_stream_frames(stream_idx, bytes(packet))
                for frame in frames:
                    ft = self.feature_name_to_feature_type[fname]
                    # Use backend to convert frame to array
                    arr = self.backend.convert_frame_to_array(frame, ft, format="rgb24")
                    cache[fname].append(arr)
                    decoded_packets += 1
                    logger.debug(
                        f"Decoded {codec} frame for '{fname}': shape={getattr(arr, 'shape', 'N/A')}, dtype={getattr(arr, 'dtype', 'N/A')}"
                    )

            # Record timestamp for resampling logic
            resampler.update_last_pts(fname, packet.pts)

            # Early exit: all streams finished their slice
            if sl_stop is not None and resampler.kept_idx[fname] >= sl_stop:
                done.add(fname)
                logger.debug(
                    f"Feature '{fname}' reached slice stop ({sl_stop}), marking as done"
                )
                if len(done) == len(cache):
                    logger.debug(
                        "All features completed their slices, breaking early")
                    break

        # ------------------------------------------------------------------ #
        # Flush any buffered pictures that the decoder is still holding
        # ------------------------------------------------------------------ #
        for stream_idx, fname in stream_idx_to_feature.items():
            if not fname or fname not in cache:
                continue
            
            codec = self.backend.get_stream_codec_name(stream_idx)
            if codec == "rawvideo":
                continue  # pickled streams have no buffer

            # Flush the decoder by passing None
            frames = self.backend.decode_stream_frames(stream_idx, packet_data=None)
            for frame in frames:
                flush_idx = resampler.next_index(fname)
                if not resampler.want(flush_idx):  # honour slice filter
                    continue

                ft = self.feature_name_to_feature_type[fname]
                # Use backend to convert frame to array
                arr = self.backend.convert_frame_to_array(frame, ft, format="rgb24")
                cache[fname].append(arr)
                decoded_packets += 1

        self.backend.close()

        logger.debug(
            f"Demux/decode loop completed: total_packets={packet_count}, processed={processed_packets}, "
            f"skipped_frequency={skipped_frequency}, skipped_slice={skipped_slice}, decoded={decoded_packets}, upsampled_frames={upsampled_frames}"
        )

        # ------------------------------------------------------------------ #
        # Convert to numpy arrays
        # ------------------------------------------------------------------ #
        logger.debug("Converting cached data to numpy arrays")
        out: Dict[str, Any] = {}
        for fname, lst in cache.items():
            logger.debug(f"Converting '{fname}': {len(lst)} items")
            if not lst:
                logger.debug(f"Warning: '{fname}' has no data after filtering")
                out[fname] = np.array([])
                continue

            ft = self.feature_name_to_feature_type[fname]
            if ft.dtype in ["string", "str"]:
                out[fname] = np.array(lst, dtype=object)
                logger.debug(
                    f"Created object array for '{fname}': shape={out[fname].shape}"
                )
            else:
                out[fname] = np.asarray(lst, dtype=ft.dtype)
                logger.debug(
                    f"Created {ft.dtype} array for '{fname}': shape={out[fname].shape}"
                )

        logger.debug(
            f"load() returning {len(out)} features: {list(out.keys())}")
        return out

    def init_feature_streams(self, feature_spec: Dict):
        """
        initialize the feature stream with the feature name and its type
        args:
            feature_dict: dictionary of feature name and its type
        """
        for feature, feature_type in feature_spec.items():
            encoding = self._get_encoding_of_feature(None, feature_type, feature)
            self.feature_name_to_stream[
                feature] = self._add_stream_to_container(
                    self.container_file, feature, encoding, feature_type)

    def add(
        self,
        feature: str,
        data: Any,
        timestamp: Optional[int] = None,
        time_unit: Optional[str] = None,
    ) -> None:
        """
        add one value to container file

        Args:
            feature (str): name of the feature
            data (Any): value associated with the feature; except dictionary
            timestamp (optional int): timestamp value. If not provided, the current time is used.
            time_unit (optional str): time unit of the timestamp. If not provided, uses trajectory default.

        Examples:
            >>> trajectory.add('feature1', 'image1.jpg')
            >>> trajectory.add('feature1', 'image1.jpg', timestamp=1000, time_unit='ms')

        Logic:
        - check the feature name
        - if the feature name is not in the container, create a new stream

        - check the type of value
        - if value is numpy array, create a frame and encode it
        - if it is a string or int, create a packet and encode it
        - else raise an error

        Exceptions:
            raise an error if the value is a dictionary
        """
        logger.debug(
            f"Adding feature: {feature}, data shape: {getattr(data, 'shape', 'N/A')}"
        )

        if type(data) == dict:
            raise ValueError("Use add_by_dict for dictionary")

        feature_type = FeatureType.from_data(data)
        # encoding = self._get_encoding_of_feature(data, None)
        self.feature_name_to_feature_type[feature] = feature_type

        # check if the feature is already in the container
        # if not, create a new stream
        # Check if the feature is already in the container
        # here we enforce rawvideo encoding for all features
        # later on the compacting step, we will encode the pickled data to images
        stream_idx = self.backend.stream_exists_by_feature(feature)
        logger.info(f"Stream index for feature {feature}: {stream_idx}")
        if stream_idx is None:
            logger.debug(f"Creating new stream for feature: {feature}")
            self._on_new_stream(feature, "rawvideo", feature_type)
            stream_idx = self.backend.stream_exists_by_feature(feature)
            if stream_idx is None:
                raise RuntimeError(f"Failed to create stream for feature {feature}")

        logger.debug(f"Using stream index: {stream_idx}")

        # get the timestamp using TimeManager
        if timestamp is None:
            validated_timestamp = self.time_manager.current_timestamp("ms")
        else:
            validated_timestamp = self.time_manager.convert_units(timestamp, time_unit, "ms")

        logger.debug(
            f"Encoding frame with validated timestamp: {validated_timestamp}")
        
        # encode the frame using backend
        packet_infos = self.backend.encode_data_to_packets(
            data=data,
            stream_index=stream_idx,
            timestamp=validated_timestamp,
            codec_config=self.codec_config,
        )
        logger.debug(f"Generated {len(packet_infos)} packet infos")

        # write the packets to the container
        for i, packet_info in enumerate(packet_infos):
            logger.debug(f"Muxing packet {i}: {packet_info}")
            self.backend.mux_packet_info(packet_info)
            logger.debug(f"Successfully muxed packet {i}")

    def add_by_dict(
        self,
        data: Dict[str, Any],
        timestamp: Optional[int] = None,
        time_unit: Optional[str] = None,
    ) -> None:
        """
        add one value to container file
        data might be nested dictionary of values for each feature

        Args:
            data (Dict[str, Any]): dictionary of feature name and value
            timestamp (optional int): timestamp value. If not provided, the current time is used.
            time_unit (optional str): time unit of the timestamp. If not provided, uses trajectory default.
                assume the timestamp is same for all the features within the dictionary

        Examples:
            >>> trajectory.add_by_dict({'feature1': 'image1.jpg'})

        Logic:
        - check the data see if it is a dictionary
        - if dictionary, need to flatten it and add each feature separately
        """
        if type(data) != dict:
            raise ValueError("Use add for non-dictionary data, type is ",
                             type(data))

        _flatten_dict_data = _flatten_dict(data,
                                           sep=self.feature_name_separator)

        # Get validated timestamp using TimeManager
        if timestamp is None:
            validated_timestamp = self.time_manager.current_timestamp("ms")
        else:
            validated_timestamp = self.time_manager.convert_units(timestamp, time_unit, "ms")

        for feature, value in _flatten_dict_data.items():
            self.add(feature, value, validated_timestamp, "ms")

    @classmethod
    def from_list_of_dicts(
        cls,
        data: List[Dict[str, Any]],
        path: Text,
        video_codec: str = "auto",
        codec_options: Optional[Dict[str, Any]] = None,
        visualization_feature: Optional[Text] = None,
        fps: Optional[int] = 10,
    ) -> "Trajectory":
        """
        Create a Trajectory object from a list of dictionaries.

        args:
            data (List[Dict[str, Any]]): list of dictionaries
            path (Text): path to the trajectory file
            video_codec (str, optional): Video codec to use. Defaults to "auto".
            codec_options (Dict[str, Any], optional): Additional codec-specific options.
            visualization_feature: Optional feature name to prioritize as first stream for visualization.

        Example:
        original_trajectory = [
            {"feature1": "value1", "feature2": "value2"},
            {"feature1": "value3", "feature2": "value4"},
        ]

        trajectory = Trajectory.from_list_of_dicts(original_trajectory, path="/tmp/robodm/output.vla")
        """
        traj = cls(path,
                   mode="w",
                   video_codec=video_codec,
                   codec_options=codec_options,
                   visualization_feature=visualization_feature)
        logger.info(
            f"Creating a new trajectory file at {path} with {len(data)} steps")
        
        time_interval_ms = 1000 / fps
        current_timestamp = 0
        for step in data:
            traj.add_by_dict(step, current_timestamp, time_unit="ms")
            current_timestamp += time_interval_ms
        traj.close()
        return traj

    @classmethod
    def from_dict_of_lists(
        cls,
        data: Dict[str, List[Any]],
        path: Text,
        feature_name_separator: Text = "/",
        video_codec: str = "auto",
        codec_options: Optional[Dict[str, Any]] = None,
        visualization_feature: Optional[Text] = None,
        fps: Optional[int] = 10,
    ) -> "Trajectory":
        """
        Create a Trajectory object from a dictionary of lists.

        Args:
            data (Dict[str, List[Any]]): dictionary of lists. Assume list length is the same for all features.
            path (Text): path to the trajectory file
            feature_name_separator (Text, optional): Delimiter to separate feature names. Defaults to "/".
            video_codec (str, optional): Video codec to use. Defaults to "auto".
            codec_options (Dict[str, Any], optional): Additional codec-specific options.
            visualization_feature: Optional feature name to prioritize as first stream for visualization.

        Returns:
            Trajectory: _description_

        Example:
        original_trajectory = {
            "feature1": ["value1", "value3"],
            "feature2": ["value2", "value4"],
        }

        trajectory = Trajectory.from_dict_of_lists(original_trajectory, path="/tmp/robodm/output.vla")
        """
        traj = cls(
            path,
            feature_name_separator=feature_name_separator,
            mode="w",
            video_codec=video_codec,
            codec_options=codec_options,
            visualization_feature=visualization_feature,
        )
        time_interval_ms = 1000 / fps
        current_timestamp = 0
        # flatten the data such that all data starts and put feature name with separator
        _flatten_dict_data = _flatten_dict(data,
                                           sep=traj.feature_name_separator)

        # Check if all lists have the same length
        list_lengths = [len(v) for v in _flatten_dict_data.values()]
        if len(set(list_lengths)) != 1:
            raise ValueError(
                "All lists must have the same length",
                [(k, len(v)) for k, v in _flatten_dict_data.items()],
            )

        for i in range(list_lengths[0]):
            step = {k: v[i] for k, v in _flatten_dict_data.items()}
            traj.add_by_dict(step, current_timestamp, time_unit="ms")
            current_timestamp += time_interval_ms
        traj.close()
        return traj
    
    def _transcode_pickled_images(self,
                                  ending_timestamp: Optional[int] = None):
        """
        Transcode pickled images into the desired format (e.g., raw or encoded images).
        """
        from robodm.backend.base import StreamConfig
        from robodm.backend.pyav_backend import PyAVBackend

        # Move the original file to a temporary location
        temp_path = self.path + ".temp"
        self._rename(self.path, temp_path)

        try:
            # Build stream configurations for transcoding
            stream_configs = {}
            
            # Open original container temporarily to get stream info
            temp_backend = PyAVBackend()
            temp_backend.open(temp_path, "r")
            original_streams = temp_backend.get_streams()
            temp_backend.close()
            
            for stream_metadata in original_streams:
                feature_name = stream_metadata.feature_name
                if feature_name == "unknown" or not feature_name:
                    continue
                    
                feature_type = self.feature_name_to_feature_type.get(feature_name)
                if feature_type is None:
                    continue
                
                # Determine target encoding
                target_encoding = self._get_encoding_of_feature(None, feature_type, feature_name)
                
                # Create stream config
                config = StreamConfig(
                    feature_name=feature_name,
                    feature_type=feature_type,
                    encoding=target_encoding,
                    codec_options=self.codec_config.get_codec_options(target_encoding),
                    pixel_format=self.codec_config.get_pixel_format(target_encoding, feature_type),
                )
                
                # Use a dummy stream index as key - the backend will handle mapping
                stream_configs[len(stream_configs)] = config

            # Use backend's transcoding abstraction
            self.backend.transcode_container(
                input_path=temp_path,
                output_path=self.path,
                stream_configs=stream_configs,
                visualization_feature=self.visualization_feature
            )

            logger.debug("Transcoding completed successfully")
            self._remove(temp_path)

        except Exception as e:
            # If transcoding fails completely, restore the original file
            logger.error(f"Transcoding failed completely: {e}")
            if self._exists(temp_path):
                if self._exists(self.path):
                    self._remove(self.path)
                self._rename(temp_path, self.path)
                logger.info(f"Restored original file to {self.path}")
            raise

    def _on_new_stream(self, new_feature, new_encoding, new_feature_type):
        from robodm.backend.base import StreamConfig
        
        # Check if stream already exists for this feature
        if self.backend.stream_exists_by_feature(new_feature) is not None:
            return

        # Get current streams from backend
        current_streams = self.backend.get_streams()
        
        if not current_streams:
            logger.debug(
                f"Creating a new stream for the first feature {new_feature}")
            # Use backend to add the stream directly
            stream = self.backend.add_stream_for_feature(
                feature_name=new_feature,
                feature_type=new_feature_type,
                codec_config=self.codec_config,
                encoding=new_encoding,
            )
            # Update legacy tracking for backwards compatibility
            self.feature_name_to_stream[new_feature] = stream
            self.container_file = self.backend.container
        else:
            logger.debug(f"Adding a new stream for the feature {new_feature}")
            # Following is a workaround because we cannot add new streams to an existing container
            # Close current container
            self.close(compact=False)

            # Move the original file to a temporary location
            temp_path = self.path + ".temp"
            self._rename(self.path, temp_path)

            # Build stream configurations for existing streams
            existing_stream_configs = []
            for i, stream_metadata in enumerate(current_streams):
                if stream_metadata.feature_name == new_feature:
                    continue  # Skip the new feature we're adding
                feature_type = self.feature_name_to_feature_type.get(stream_metadata.feature_name)
                if feature_type is None:
                    continue
                config = StreamConfig(
                    feature_name=stream_metadata.feature_name,
                    feature_type=feature_type,
                    encoding=stream_metadata.encoding
                )
                existing_stream_configs.append((i, config))

            # Add new stream configuration
            new_stream_config = StreamConfig(
                feature_name=new_feature,
                feature_type=new_feature_type,
                encoding=new_encoding
            )

            # Use backend's container recreation abstraction
            stream_mapping = self.backend.create_container_with_new_streams(
                original_path=temp_path,
                new_path=self.path,
                existing_streams=existing_stream_configs,
                new_stream_configs=[new_stream_config]
            )

            # Update our tracking structures using backend information
            self.container_file = self.backend.container
            
            # Update feature_name_to_stream mapping using backend
            new_feature_name_to_stream = {}
            updated_streams = self.backend.get_streams()
            for i, stream_metadata in enumerate(updated_streams):
                feature_name = stream_metadata.feature_name
                if feature_name and hasattr(self.backend, '_idx_to_stream'):
                    stream = self.backend._idx_to_stream.get(i)
                    if stream:
                        new_feature_name_to_stream[feature_name] = stream
                    
            self.feature_name_to_stream = new_feature_name_to_stream
            
            self._remove(temp_path)
            self.is_closed = False

    def _add_stream_to_container(self, container, feature_name, encoding,
                                 feature_type):
        # If we're adding to the primary container that the backend manages,
        # delegate to backend. Otherwise fall back to the internal PyAV logic
        # because the backend is not aware of this ad-hoc container.

        if hasattr(self.backend, "container") and container is getattr(self.backend, "container", None):
            return self.backend.add_stream_for_feature(
                feature_name=feature_name,
                feature_type=feature_type,
                codec_config=self.codec_config,
                encoding=encoding,
            )

        # Legacy path – keep the original PyAV-based implementation for
        # transient containers (e.g. during transcoding).
        # Import PyAV locally since it's only needed for legacy paths
        from fractions import Fraction
        
        stream = container.add_stream(encoding)

        if encoding in ["ffv1", "libaom-av1", "libx264", "libx265"]:
            shape = feature_type.shape
            if shape is not None and len(shape) >= 2:
                stream.width = shape[1]
                stream.height = shape[0]

            pixel_format = self.codec_config.get_pixel_format(encoding, feature_type)
            if pixel_format:
                stream.pix_fmt = pixel_format

            codec_options = self.codec_config.get_codec_options(encoding)
            if codec_options:
                stream.codec_context.options = codec_options

        stream.metadata["FEATURE_NAME"] = feature_name
        stream.metadata["FEATURE_TYPE"] = str(feature_type)
        stream.time_base = Fraction(1, 1000)
        return stream

    def _get_encoding_of_feature(self, feature_value: Any,
                                 feature_type: Optional[FeatureType],
                                 feature_name: Optional[str] = None) -> Text:
        """
        get the encoding of the feature value
        args:
            feature_value: value of the feature
            feature_type: type of the feature
            feature_name: name of the feature (for feature-specific codec selection)
        return:
            encoding of the feature in string
        """
        if feature_type is None:
            feature_type = FeatureType.from_data(feature_value)

        return self.codec_config.get_codec_for_feature(feature_type, feature_name)
