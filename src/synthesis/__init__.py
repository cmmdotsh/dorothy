"""Story synthesis module for Dorothy."""

from src.synthesis.llm_client import LLMClient
from src.synthesis.summarizer import StorySummarizer, SynthesizedStory

__all__ = ["LLMClient", "StorySummarizer", "SynthesizedStory"]
