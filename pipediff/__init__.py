"""Top-level package for pipediff."""
from pipediff.column_diff import ColumnDiff
from pipediff.index_diff import IndexDiff
from pipediff.frame_diff import FrameDiff
from pipediff.diff_tracker import DiffTracker

__all__ = ("ColumnDiff", "IndexDiff", "FrameDiff", "DiffTracker")

__author__ = """Adem Frenk"""
__email__ = "adem.frenk@gmail.com"
__version__ = "0.0.1"
