"""Embedding generation module for Dorothy."""

from src.embeddings.client import EmbeddingClient
from src.embeddings.generator import generate_embeddings, generate_embeddings_for_articles

__all__ = ["EmbeddingClient", "generate_embeddings", "generate_embeddings_for_articles"]
