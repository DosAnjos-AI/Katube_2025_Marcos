"""
Katube Core Module - CPU-First Architecture
Base classes and resource management for modular pipeline
"""

from .resource_manager import ResourceManager
from .base_processor import BaseProcessor

__all__ = ['ResourceManager', 'BaseProcessor']
