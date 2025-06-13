"""
Test cases for robodm TimeManager system.

Tests cover:
- Time unit conversions
- Monotonic timestamp enforcement
- Datetime handling and conversions
- Integration with Trajectory class
- Edge cases and error handling
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from robodm import Trajectory
from robodm.trajectory import TimeManager


class TestTimeManager:
    """Test the TimeManager class functionality."""

    def test_time_unit_conversions(self):
        """Test conversion between different time units."""
        tm = TimeManager(time_unit="ms")

        # Test conversion to nanoseconds
        assert tm.convert_to_nanoseconds(1000, "ms") == 1_000_000_000
        assert tm.convert_to_nanoseconds(1, "s") == 1_000_000_000
        assert tm.convert_to_nanoseconds(1000, "μs") == 1_000_000
        assert tm.convert_to_nanoseconds(1000, "ns") == 1000

        # Test conversion from nanoseconds
        assert tm.convert_from_nanoseconds(1_000_000_000, "ms") == 1000
        assert tm.convert_from_nanoseconds(1_000_000_000, "s") == 1
        assert tm.convert_from_nanoseconds(1_000_000, "μs") == 1000
        assert tm.convert_from_nanoseconds(1000, "ns") == 1000

        # Test unit conversion
        assert tm.convert_units(1, "s", "ms") == 1000
        assert tm.convert_units(1000, "ms", "s") == 1
        assert tm.convert_units(1000, "μs", "ms") == 1

    def test_invalid_time_units(self):
        """Test handling of invalid time units."""
        with pytest.raises(ValueError):
            TimeManager(time_unit="invalid")

        tm = TimeManager()
        with pytest.raises(ValueError):
            tm.convert_to_nanoseconds(1000, "invalid")

        with pytest.raises(ValueError):
            tm.convert_from_nanoseconds(1000, "invalid")

    def test_datetime_conversions(self):
        """Test datetime to timestamp conversions."""
        base_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        tm = TimeManager(base_datetime=base_dt, time_unit="ms")

        # Test conversion of datetime 1 hour after base
        test_dt = base_dt + timedelta(hours=1)
        timestamp_ms = tm.datetime_to_timestamp(test_dt, "ms")
        assert timestamp_ms == 3600 * 1000  # 1 hour in milliseconds

        # Test reverse conversion
        converted_dt = tm.timestamp_to_datetime(timestamp_ms, "ms")
        assert converted_dt == test_dt

        # Test with different time units
        timestamp_s = tm.datetime_to_timestamp(test_dt, "s")
        assert timestamp_s == 3600  # 1 hour in seconds

    def test_monotonic_enforcement(self):
        """Test monotonic timestamp enforcement."""
        tm = TimeManager(time_unit="ms", enforce_monotonic=True)

        # First timestamp should pass through
        ts1 = tm.validate_timestamp(1000)
        assert ts1 == 1000

        # Second timestamp should be adjusted if not monotonic
        ts2 = tm.validate_timestamp(500)  # Earlier than previous
        assert ts2 > ts1

        # Valid monotonic timestamp should pass through
        ts3 = tm.validate_timestamp(2000)
        assert ts3 == 2000

    def test_non_monotonic_mode(self):
        """Test behavior when monotonic enforcement is disabled."""
        tm = TimeManager(time_unit="ms", enforce_monotonic=False)

        ts1 = tm.validate_timestamp(1000)
        assert ts1 == 1000

        # Should allow non-monotonic timestamps
        ts2 = tm.validate_timestamp(500)
        assert ts2 == 500

    def test_add_timestep(self):
        """Test adding timesteps to current timestamp."""
        tm = TimeManager(time_unit="ms")

        # First timestep
        ts1 = tm.add_timestep(100)  # 100ms
        assert ts1 == 100

        # Second timestep should be cumulative
        ts2 = tm.add_timestep(50)  # +50ms
        assert ts2 == 150

        # Test with different units
        ts3 = tm.add_timestep(1, "s")  # +1 second = +1000ms
        assert ts3 == 1150

    def test_create_timestamp_sequence(self):
        """Test creating sequences of monotonic timestamps."""
        tm = TimeManager(time_unit="ms", enforce_monotonic=False
                         )  # Disable monotonic for predictable sequences

        timestamps = tm.create_timestamp_sequence(
            start_timestamp=0,
            count=5,
            timestep=100  # 100ms steps
        )

        expected = [0, 100, 200, 300, 400]
        assert timestamps == expected

        # Test with different units (reset TimeManager)
        tm2 = TimeManager(time_unit="ms", enforce_monotonic=False)
        timestamps_s = tm2.create_timestamp_sequence(start_timestamp=0,
                                                     count=3,
                                                     timestep=1,
                                                     unit="s")

        expected_s = [0, 1000, 2000]  # Converted to milliseconds
        assert timestamps_s == expected_s

    def test_reset_functionality(self):
        """Test resetting the TimeManager state."""
        tm = TimeManager(time_unit="ms")

        # Add some timestamps
        tm.validate_timestamp(1000)
        tm.validate_timestamp(2000)

        # Reset should clear internal state
        new_base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        tm.reset(base_datetime=new_base)

        # Should be able to use earlier timestamps after reset
        ts = tm.validate_timestamp(500)
        assert ts == 500

    def test_timezone_handling(self):
        """Test proper timezone handling in datetime conversions."""
        # Test with UTC timezone
        base_dt_utc = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        tm_utc = TimeManager(base_datetime=base_dt_utc)

        # Test with different timezone
        base_dt_est = datetime(2023,
                               1,
                               1,
                               7,
                               0,
                               0,
                               tzinfo=timezone(timedelta(hours=-5)))  # EST
        tm_est = TimeManager(base_datetime=base_dt_est)

        # Both should give same result for same absolute time
        test_dt_utc = base_dt_utc + timedelta(hours=1)
        test_dt_est = base_dt_est + timedelta(hours=1)

        ts_utc = tm_utc.datetime_to_timestamp(test_dt_utc)
        ts_est = tm_est.datetime_to_timestamp(test_dt_est)

        assert ts_utc == ts_est  # Should be the same relative to their bases


class TestTrajectoryTimeIntegration:
    """Test integration of TimeManager with Trajectory class."""

    def test_trajectory_with_time_manager(self):
        """Test that Trajectory properly uses TimeManager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "test_trajectory.mkv")
            base_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

            # Create trajectory with specific time settings
            trajectory = Trajectory(
                path,
                mode="w",
                base_datetime=base_dt,
                time_unit="ms",
                enforce_monotonic=True,
            )

            # Add data with explicit timestamps
            trajectory.add("feature1",
                           "value1",
                           timestamp=1000,
                           time_unit="ms")
            trajectory.add("feature1",
                           "value2",
                           timestamp=2000,
                           time_unit="ms")
            trajectory.add("feature1",
                           "value3",
                           timestamp=1500,
                           time_unit="ms")  # Should be adjusted

            trajectory.close()

            # Load and verify
            trajectory_read = Trajectory(path, mode="r")
            data = trajectory_read.load()
            trajectory_read.close()

            assert len(data["feature1"]) == 3

    def test_trajectory_datetime_based_timestamps(self):
        """Test trajectory with datetime-based timestamp calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "test_trajectory.mkv")
            base_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

            trajectory = Trajectory(path,
                                   mode="w",
                                   base_datetime=base_dt,
                                   time_unit="ms")

            # Add data at specific datetime points
            dt1 = base_dt + timedelta(seconds=1)
            dt2 = base_dt + timedelta(seconds=2)

            ts1 = trajectory.time_manager.datetime_to_timestamp(dt1, "ms")
            ts2 = trajectory.time_manager.datetime_to_timestamp(dt2, "ms")

            trajectory.add("sensor1", 100.0, timestamp=ts1, time_unit="ms")
            trajectory.add("sensor1", 200.0, timestamp=ts2, time_unit="ms")

            trajectory.close()

            # Verify timestamps are as expected
            assert ts1 == 1000  # 1 second = 1000ms
            assert ts2 == 2000  # 2 seconds = 2000ms

    def test_trajectory_auto_timestamps(self):
        """Test trajectory with automatic timestamp generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "test_trajectory.mkv")

            trajectory = Trajectory(path, mode="w", time_unit="ms")

            # Add data without explicit timestamps
            trajectory.add("feature1", "value1")
            trajectory.add("feature1", "value2")
            trajectory.add("feature1", "value3")

            trajectory.close()

            # Should create trajectory without errors
            trajectory_read = Trajectory(path, mode="r")
            data = trajectory_read.load()
            trajectory_read.close()

            assert len(data["feature1"]) == 3

    def test_trajectory_mixed_time_units(self):
        """Test trajectory with mixed time units in different add() calls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "test_trajectory.mkv")

            trajectory = Trajectory(path, mode="w", time_unit="ms")

            # Add data with different time units
            trajectory.add("sensor1", 1.0, timestamp=1,
                           time_unit="s")  # 1000ms
            trajectory.add("sensor1", 2.0, timestamp=1500,
                           time_unit="ms")  # 1500ms
            trajectory.add("sensor1", 3.0, timestamp=2000000,
                           time_unit="μs")  # 2000ms

            trajectory.close()

            trajectory_read = Trajectory(path, mode="r")
            data = trajectory_read.load()
            trajectory_read.close()

            assert len(data["sensor1"]) == 3


class TestTimeManagerEdgeCases:
    """Test edge cases and error conditions."""

    def test_large_timestamp_values(self):
        """Test handling of very large timestamp values."""
        tm = TimeManager(time_unit="ns")

        # Test nanosecond precision with large values
        large_ns = 9223372036854775807  # Near max int64
        ts_ms = tm.convert_from_nanoseconds(large_ns, "ms")
        back_to_ns = tm.convert_to_nanoseconds(ts_ms, "ms")

        # Should handle large values without overflow
        assert isinstance(ts_ms, int)
        assert isinstance(back_to_ns, int)

    def test_zero_and_negative_timestamps(self):
        """Test handling of zero and negative timestamp values."""
        tm = TimeManager(time_unit="ms", enforce_monotonic=False)

        # Should handle zero timestamps
        ts = tm.validate_timestamp(0)
        assert ts == 0

        # Should handle negative timestamps when monotonic is disabled
        ts_neg = tm.validate_timestamp(-1000)
        assert ts_neg == -1000

    def test_floating_point_timestamps(self):
        """Test handling of floating point timestamp inputs."""
        tm = TimeManager(time_unit="ms")

        # Should handle float inputs by converting to int
        ts = tm.validate_timestamp(1500.7)
        assert isinstance(ts, int)
        assert ts == 1500

        # Test float conversion in timestep
        ts_step = tm.add_timestep(100.5)
        assert isinstance(ts_step, int)

    def test_sequence_with_overlap_handling(self):
        """Test timestamp sequence generation with overlap scenarios."""
        tm = TimeManager(time_unit="ms", enforce_monotonic=True)

        # Set initial state
        tm.validate_timestamp(5000)

        # Create sequence that would overlap with existing state
        timestamps = tm.create_timestamp_sequence(
            start_timestamp=3000,
            count=3,
            timestep=1000  # Earlier than current state
        )

        # Should adjust to maintain monotonic ordering
        assert all(ts > 5000 for ts in timestamps)
        assert timestamps[1] > timestamps[0]
        assert timestamps[2] > timestamps[1]


class TestTimeManagerPerformance:
    """Test performance characteristics of TimeManager."""

    def test_large_timestamp_sequence_generation(self):
        """Test generating large sequences of timestamps efficiently."""
        tm = TimeManager(
            time_unit="ms",
            enforce_monotonic=False)  # Disable for predictable sequence

        # Generate large sequence
        timestamps = tm.create_timestamp_sequence(start_timestamp=0,
                                                  count=10000,
                                                  timestep=1)

        assert len(timestamps) == 10000
        assert timestamps[0] == 0
        assert timestamps[-1] == 9999

        # Verify monotonic ordering
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

    def test_many_timestamp_validations(self):
        """Test performance of many timestamp validations."""
        tm = TimeManager(time_unit="ms", enforce_monotonic=True)

        # Validate many timestamps
        timestamps = []
        for i in range(1000):
            ts = tm.validate_timestamp(i)
            timestamps.append(ts)

        # Should maintain monotonic ordering
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
