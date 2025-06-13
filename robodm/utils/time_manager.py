

from datetime import datetime, timedelta, timezone
from fractions import Fraction
from typing import Optional, Union, List
import time
import av
import logging
logger = logging.getLogger(__name__)

class TimeManager:
    """
    Comprehensive time management system for robodm trajectories.

    Handles:
    - Multiple time units (nanoseconds, microseconds, milliseconds, seconds)
    - Base datetime reference points
    - Monotonic timestamp enforcement
    - Unit conversions
    - Per-timestep timing from base datetime
    """

    # Time unit conversion factors to nanoseconds
    TIME_UNITS = {
        "ns": 1,
        "nanoseconds": 1,
        "μs": 1_000,
        "us": 1_000,
        "microseconds": 1_000,
        "ms": 1_000_000,
        "milliseconds": 1_000_000,
        "s": 1_000_000_000,
        "seconds": 1_000_000_000,
    }

    # Trajectory time base (for robodm compatibility)
    TRAJECTORY_TIME_BASE = Fraction(1, 1000)  # milliseconds

    def __init__(
        self,
        base_datetime: Optional[datetime] = None,
        time_unit: str = "ms",
        enforce_monotonic: bool = True,
    ):
        """
        Initialize TimeManager.

        Parameters:
        -----------
        base_datetime : datetime, optional
            Reference datetime for relative timestamps. If None, uses current time.
        time_unit : str
            Default time unit for timestamp inputs ('ns', 'μs', 'ms', 's')
        enforce_monotonic : bool
            Whether to enforce monotonically increasing timestamps
        """
        self.base_datetime = base_datetime or datetime.now(timezone.utc)
        self.time_unit = time_unit
        self.enforce_monotonic = enforce_monotonic

        # Internal state
        self._last_timestamp_ns = 0
        self._start_time = time.time()

        # Validate time unit
        if time_unit not in self.TIME_UNITS:
            raise ValueError(f"Unsupported time unit: {time_unit}. "
                             f"Supported: {list(self.TIME_UNITS.keys())}")

    def reset(self, base_datetime: Optional[datetime] = None):
        """Reset the time manager with new base datetime."""
        if base_datetime:
            self.base_datetime = base_datetime
        self._last_timestamp_ns = 0
        self._start_time = time.time()

    def current_timestamp(self, unit: Optional[str] = None) -> int:
        """
        Get current timestamp relative to start time.

        Parameters:
        -----------
        unit : str, optional
            Time unit for returned timestamp. If None, uses default unit.

        Returns:
        --------
        int : Current timestamp in specified unit
        """
        unit = unit or self.time_unit
        current_time_ns = int((time.time() - self._start_time) * 1_000_000_000)
        return self.convert_from_nanoseconds(current_time_ns, unit)

    def datetime_to_timestamp(self,
                              dt: datetime,
                              unit: Optional[str] = None) -> int:
        """
        Convert datetime to timestamp relative to base_datetime.

        Parameters:
        -----------
        dt : datetime
            Datetime to convert
        unit : str, optional
            Target time unit. If None, uses default unit.

        Returns:
        --------
        int : Timestamp in specified unit
        """
        unit = unit or self.time_unit
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if self.base_datetime.tzinfo is None:
            base_dt = self.base_datetime.replace(tzinfo=timezone.utc)
        else:
            base_dt = self.base_datetime

        delta_seconds = (dt - base_dt).total_seconds()
        delta_ns = int(delta_seconds * 1_000_000_000)
        return self.convert_from_nanoseconds(delta_ns, unit)

    def timestamp_to_datetime(self,
                              timestamp: int,
                              unit: Optional[str] = None) -> datetime:
        """
        Convert timestamp to datetime using base_datetime as reference.

        Parameters:
        -----------
        timestamp : int
            Timestamp value
        unit : str, optional
            Time unit of input timestamp. If None, uses default unit.

        Returns:
        --------
        datetime : Corresponding datetime
        """
        unit = unit or self.time_unit
        timestamp_ns = self.convert_to_nanoseconds(timestamp, unit)
        delta_seconds = timestamp_ns / 1_000_000_000

        if self.base_datetime.tzinfo is None:
            base_dt = self.base_datetime.replace(tzinfo=timezone.utc)
        else:
            base_dt = self.base_datetime

        return base_dt + timedelta(seconds=delta_seconds)

    def convert_to_nanoseconds(self, timestamp: Union[int, float],
                               unit: str) -> int:
        """Convert timestamp from given unit to nanoseconds."""
        if unit not in self.TIME_UNITS:
            raise ValueError(f"Unsupported time unit: {unit}")
        return int(timestamp * self.TIME_UNITS[unit])

    def convert_from_nanoseconds(self, timestamp_ns: int, unit: str) -> int:
        """Convert timestamp from nanoseconds to given unit."""
        if unit not in self.TIME_UNITS:
            raise ValueError(f"Unsupported time unit: {unit}")
        return int(timestamp_ns // self.TIME_UNITS[unit])

    def convert_units(self, timestamp: Union[int, float], from_unit: str,
                      to_unit: str) -> int:
        """Convert timestamp between different units."""
        timestamp_ns = self.convert_to_nanoseconds(timestamp, from_unit)
        return self.convert_from_nanoseconds(timestamp_ns, to_unit)

    def add_timestep(self,
                     timestep: Union[int, float],
                     unit: Optional[str] = None) -> int:
        """
        Add a timestep to the last timestamp and return trajectory-compatible timestamp.

        Parameters:
        -----------
        timestep : int or float
            Time step to add
        unit : str, optional
            Time unit of timestep

        Returns:
        --------
        int : New timestamp in trajectory time base units (milliseconds)
        """
        unit = unit or self.time_unit
        timestep_ns = self.convert_to_nanoseconds(timestep, unit)
        new_timestamp_ns = self._last_timestamp_ns + timestep_ns

        self._last_timestamp_ns = new_timestamp_ns
        return self.convert_from_nanoseconds(new_timestamp_ns, "ms")

    def create_timestamp_sequence(
        self,
        start_timestamp: int,
        count: int,
        timestep: Union[int, float],
        unit: Optional[str] = None,
    ) -> List[int]:
        """
        Create a sequence of monotonic timestamps.

        Parameters:
        -----------
        start_timestamp : int
            Starting timestamp
        count : int
            Number of timestamps to generate
        timestep : int or float
            Time step between consecutive timestamps
        unit : str, optional
            Time unit for inputs

        Returns:
        --------
        List[int] : List of timestamps in trajectory time base units
        """
        unit = unit or self.time_unit
        start_ns = self.convert_to_nanoseconds(start_timestamp, unit)
        timestep_ns = self.convert_to_nanoseconds(timestep, unit)

        timestamps = []
        current_ns = start_ns

        for i in range(count):
            # Ensure monotonic ordering if enforce_monotonic is True
            if self.enforce_monotonic and current_ns <= self._last_timestamp_ns:
                current_ns = self._last_timestamp_ns + 1_000_000  # +1ms in nanoseconds

            timestamps.append(self.convert_from_nanoseconds(current_ns, "ms"))

            # Update last timestamp only if monotonic enforcement is enabled
            if self.enforce_monotonic:
                self._last_timestamp_ns = current_ns

            current_ns += timestep_ns

        return timestamps
