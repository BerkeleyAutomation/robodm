from __future__ import annotations

"""Utility class for frequency based up-/down-sampling and slice filtering.
This logic used to live inside Trajectory.load() but was extracted so the
Trajectory class can focus on IO while this helper focuses purely on the
index accounting.
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FrequencyResampler:
    """Book-keeps per-feature indices for frequency resampling **and** slice
    filtering.

    A single instance is shared across all feature streams.  Each feature is
    registered via :py:meth:`register_feature` which initialises its internal
    bookkeeping (``kept_idx`` & ``last_pts``).

    For every incoming packet timestamp the caller invokes
    :py:meth:`process_packet` which returns a small instruction set telling the
    caller whether the current packet should be kept and how many *duplicate*
    frames (for up-sampling) need to be emitted **before** the current packet.

    The caller is responsible for actually materialising those duplicates –
    the resampler only deals with the *indices*.
    """

    def __init__(
        self,
        period_ms: Optional[int],
        sl_start: int,
        sl_stop: Optional[int],
        sl_step: int,
        seek_offset_frames: int = 0,
    ) -> None:
        self.period_ms = period_ms
        self.sl_start = sl_start
        self.sl_stop = sl_stop
        self.sl_step = sl_step
        self._seek_offset_frames = seek_offset_frames

        # Per-feature bookkeeping
        self.last_pts: Dict[str, Optional[int]] = {}
        self.kept_idx: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Registration helpers
    # ------------------------------------------------------------------ #
    def register_feature(self, fname: str) -> None:
        """Register *fname* with initial indices properly set up."""
        if fname in self.kept_idx:
            return
        # If we performed a seek() the kept_idx should start at
        # (seek_offset_frames − 1) so that the first *kept* packet receives
        # index "seek_offset_frames" (because we increment before checking).
        self.kept_idx[fname] = self._seek_offset_frames - 1
        self.last_pts[fname] = None
        logger.debug(
            "Resampler: registered feature '%s' with initial kept_idx=%d",
            fname,
            self.kept_idx[fname],
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process_packet(
        self,
        fname: str,
        pts: Optional[int],
        has_prior_frame: bool,
    ) -> tuple[bool, int]:
        """Determine whether *packet* should be kept and how many *duplicate*
        frames (if any) should be emitted *before* it.

        Parameters
        ----------
        fname
            Feature name the packet belongs to.
        pts
            Packet timestamp (milliseconds).
        has_prior_frame
            Whether the caller has already produced at least one frame for
            *fname*.  Needed so that we don't try to duplicate when we don't
            have a previous frame yet.

        Returns
        -------
        keep_current
            ``True`` if the current packet passes the frequency filter.
        num_duplicates
            Number of duplicate frames that should be emitted **before** the
            current packet to fill large temporal gaps (upsampling).  Will be
            ``0`` for down-sampling or when *period_ms* is ``None``.
        """
        if pts is None:
            # Defensive – treat missing pts like "keep" with no up-sampling.
            logger.debug("Resampler: packet for '%s' has no pts – keeping.", fname)
            keep_current = True
            num_duplicates = 0
        elif self.period_ms is None:
            # Resampling disabled – keep everything.
            keep_current = True
            num_duplicates = 0
        else:
            last = self.last_pts[fname]
            if last is None:
                # First packet – always keep, no duplicates necessary.
                keep_current = True
                num_duplicates = 0
            else:
                gap = pts - last
                if gap < self.period_ms:
                    # Down-sampling: skip current packet.
                    keep_current = False
                    num_duplicates = 0
                else:
                    # Keep current packet.  If the gap is big we might need to
                    # up-sample by inserting *duplicate* frames beforehand.
                    if gap > self.period_ms and has_prior_frame:
                        num_duplicates = int(gap // self.period_ms) - 1
                    else:
                        num_duplicates = 0
                    keep_current = True

        return keep_current, num_duplicates

    # ------------------------------------------------------------------ #
    # Index helpers
    # ------------------------------------------------------------------ #
    def next_index(self, fname: str) -> int:
        """Increment *kept_idx* for *fname* and return the new value."""
        self.kept_idx[fname] += 1
        return self.kept_idx[fname]

    # ------------------------------------------------------------------ #
    # Slice filtering helpers
    # ------------------------------------------------------------------ #
    def want(self, idx: int) -> bool:
        if idx < self.sl_start:
            return False
        if self.sl_stop is not None and idx >= self.sl_stop:
            return False
        return ((idx - self.sl_start) % self.sl_step) == 0

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    def update_last_pts(self, fname: str, pts: Optional[int]) -> None:
        self.last_pts[fname] = pts 