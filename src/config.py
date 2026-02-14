"""Configuration management for Dorothy."""

from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

from src.models import BiasRating, Column, FetchMethod, Source


class OpenSearchSettings(BaseSettings):
    """OpenSearch connection settings."""

    host: str = "localhost"
    port: int = 9200
    username: str = ""
    password: str = ""
    use_ssl: bool = False
    verify_certs: bool = False

    class Config:
        env_prefix = "OPENSEARCH_"


class FetcherSettings(BaseSettings):
    """RSS fetcher settings."""

    timeout: float = 30.0
    user_agent: str = "Dorothy/0.1 (news aggregator)"
    batch_size: int = 50

    class Config:
        env_prefix = "FETCHER_"


class SchedulerSettings(BaseSettings):
    """Scheduler settings."""

    fetch_interval_minutes: int = 60

    class Config:
        env_prefix = "SCHEDULER_"


class EmbeddingSettings(BaseSettings):
    """Embedding service settings."""

    base_url: str = "http://192.168.0.149:1234"
    model: str = "text-embedding-mxbai-embed-large-v1"
    batch_size: int = 32
    enabled: bool = True

    class Config:
        env_prefix = "EMBEDDING_"


class LLMSettings(BaseSettings):
    """LLM service settings for story synthesis."""

    base_url: str = "http://192.168.0.149:1234"
    model: str = "qwen/qwen3-30b-a3b-2507"
    temperature: float = 0.3
    max_tokens: int = 1500
    context_length: int = 32768

    class Config:
        env_prefix = "LLM_"


class DorothyConfig:
    """Main configuration class for Dorothy."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.opensearch = OpenSearchSettings()
        self.fetcher = FetcherSettings()
        self.scheduler = SchedulerSettings()
        self.embedding = EmbeddingSettings()
        self.llm = LLMSettings()
        self._sources: list[Source] = []

    def load_sources(self) -> list[Source]:
        """Load sources from YAML file."""
        sources_path = self.config_dir / "sources.yaml"

        if not sources_path.exists():
            raise FileNotFoundError(f"Sources config not found: {sources_path}")

        with open(sources_path) as f:
            data = yaml.safe_load(f)

        sources = []
        for src in data.get("sources", []):
            sources.append(
                Source(
                    name=src["name"],
                    slug=src["slug"],
                    rss_url=src.get("rss_url"),
                    fetch_method=FetchMethod(src["fetch_method"]),
                    column=Column(src["column"]),
                    bias=BiasRating(src["bias"]),
                    active=src.get("active", True),
                )
            )

        self._sources = sources
        return sources

    @property
    def sources(self) -> list[Source]:
        """Get loaded sources (loads if needed)."""
        if not self._sources:
            self.load_sources()
        return self._sources

    def get_active_rss_sources(self) -> list[Source]:
        """Get only active RSS sources."""
        return [s for s in self.sources if s.active and s.fetch_method == FetchMethod.RSS]


# Global config instance
config = DorothyConfig()
