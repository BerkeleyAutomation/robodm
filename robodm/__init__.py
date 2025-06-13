# robodm: A high-performance robotics data management framework
# Copyright (c) 2024 Berkeley Automation Lab

import os

__root_dir__ = os.path.dirname(os.path.abspath(__file__))

# from robodm import dataset, episode, feature
# from robodm.dataset import Dataset
# from robodm import trajectory

from robodm.feature import FeatureType
from robodm.trajectory import Trajectory
from robodm.trajectory_base import (FileSystemInterface, TimeProvider,
                                    TrajectoryInterface)

__all__ = [
    "FeatureType",
    "Trajectory",
    "TrajectoryInterface",
    "FileSystemInterface",
    "TimeProvider",
]

# Version of the robodm package
__version__ = "0.1.0"

# Metadata
__author__ = "Berkeley Automation Lab"
__email__ = "automation@berkeley.edu"
__description__ = "A high-performance robotics data management framework"
__url__ = "https://github.com/BerkeleyAutomation/robodm"
__license__ = "BSD-3-Clause"

import logging

_FORMAT = "%(levelname).1s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=_FORMAT)
logging.root.setLevel(logging.INFO)
