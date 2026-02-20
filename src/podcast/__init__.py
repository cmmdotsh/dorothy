"""Podcast generation module for Dorothy."""

from src.podcast.script_writer import ScriptWriter
from src.podcast.generator import PodcastGenerator

__all__ = ["ScriptWriter", "PodcastGenerator"]

try:
    from src.podcast.tts_client import TTSClient

    __all__.append("TTSClient")
except ImportError:
    pass
