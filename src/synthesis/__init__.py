"""Synthesis package for Dorothy."""

from src.synthesis.llm_client import LLMClient, LLMError
from src.synthesis.summarizer import StorySummarizer, SynthesizedStory

__all__ = [
    "LLMClient",
    "LLMError",
    "StorySummarizer",
    "SynthesizedStory",
]
