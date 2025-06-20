# robodm-agentic: An agentic robotics data management framework
# Copyright (c) 2025 Berkeley Automation Lab

import os

__root_dir__ = os.path.dirname(os.path.abspath(__file__))

# Version of the robodm-agentic package
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

# Core agentic components
from .core.agent import RoboDMAgent
from .core.robodm_interface import RoboDMInterface
from .mcp.server import RoboDMMCPServer
from .clients.llm_client import LLMClient
from .clients.vlm_client import VLMClient

__all__ = [
    "RoboDMAgent",
    "RoboDMInterface", 
    "RoboDMMCPServer",
    "LLMClient",
    "VLMClient",
]
