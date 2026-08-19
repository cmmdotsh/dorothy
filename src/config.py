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
    """Embedding service settings (via LMStudio)."""

    base_url: str = "http://192.168.0.149:1234"
    model: str = "text-embedding-qwen3-embedding-0.6b"
    batch_size: int = 32
    enabled: bool = True

    class Config:
        env_prefix = "EMBEDDING_"


class LLMSettings(BaseSettings):
    """LLM service settings for story synthesis (via LMStudio)."""

    base_url: str = "http://192.168.0.149:1234"
    model: str = "mlx-community/qwen3.5-35b-a3b"
    temperature: float = 0.3
    max_tokens: int = 4096
    context_length: int = 32768
    # Prepend an empty <think></think> turn to suppress reasoning traces.
    # Needed for Qwen "thinking" models; breaks models with their own chat
    # template (e.g. LiquidAI LFM2 emits tool-call tokens). Set false for those.
    skip_thinking: bool = True

    class Config:
        env_prefix = "LLM_"


class ExtractorSettings(BaseSettings):
    """Article body extraction settings."""

    enabled: bool = True
    timeout: float = 30.0
    delay: float = 1.0
    batch_size: int = 500
    max_workers: int = 10

    class Config:
        env_prefix = "EXTRACTOR_"


class ClaimGraphSettings(BaseSettings):
    """Claim graph analysis settings."""

    enabled: bool = True
    # Dedicated embedder for chunk corroboration. Chunk embeddings are
    # transient (recomputed per story, never stored), so this model can
    # differ from the global EMBEDDING_MODEL whose vectors persist in
    # OpenSearch. Empty string = fall back to the global embedding model.
    # Shootout 2026-08-17 on 108 LLM-judged chunk pairs: qwen3-embedding-0.6b
    # AUC 0.950 / F1 0.82 vs mxbai-large AUC 0.931 / F1 0.71.
    embedding_model: str = "text-embedding-qwen3-embedding-0.6b"
    # Chunk-pair cosine floor for corroboration, tuned PER MODEL:
    # qwen3-embedding-0.6b optimum 0.74 (precision 75%, recall 83%);
    # for mxbai-large use 0.82 (precision 75%, recall 67%).
    similarity_threshold: float = 0.74
    min_sources_corroborated: int = 2
    embedding_concurrency: int = 32
    min_chunk_chars: int = 80
    max_chunk_chars: int = 800

    class Config:
        env_prefix = "CLAIM_GRAPH_"


class ClusteringSettings(BaseSettings):
    """Story clustering settings."""
    # Only articles published within this window enter clustering (daily paper).
    window_hours: int = 72
    # Per-cycle cap of articles per source per column (archive-dump guard).
    max_per_source: int = 40
    min_cluster_size: int = 3
    min_samples: int = 2
    # "leaf" extracts tight leaf clusters; "eom" measurably collapses dense
    # columns (money/lifestyle) into single grab-bag blobs.
    cluster_selection_method: str = "leaf"

    class Config:
        env_prefix = "CLUSTERING_"


class EventSettings(BaseSettings):
    """Event thread matching settings."""
    # Master switch for the event-threads feature (pipeline stages guard on it).
    enabled: bool = True
    # Minimum cosine similarity for a thread/threadless candidate to reach
    # the LLM confirm shortlist.
    shortlist_threshold: float = 0.60
    # Max candidates confirmed by the LLM per story per stage.
    shortlist_k: int = 3
    # Events unseen for this many days become dormant.
    dormant_after_days: int = 14
    # How far back threadless syntheses are considered for recurrence birth.
    threadless_window_days: int = 14

    class Config:
        env_prefix = "EVENTS_"


class PodcastSettings(BaseSettings):
    """Podcast generation settings."""

    enabled: bool = False
    voice_ref_a: str = "config/voices/anchor_a.wav"
    voice_ref_b: str = "config/voices/anchor_b.wav"
    tts_device: str = "cpu"
    tts_workers: int = 1
    story_count: int = 5
    target_wpm: int = 150
    atempo: float = 1.1
    output_format: str = "mp3"
    bitrate: str = "128k"
    hf_fallback: bool = False
    hf_token: str = ""

    class Config:
        env_prefix = "PODCAST_"


class DorothyConfig:
    """Main configuration class for Dorothy."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.opensearch = OpenSearchSettings()
        self.fetcher = FetcherSettings()
        self.scheduler = SchedulerSettings()
        self.embedding = EmbeddingSettings()
        self.llm = LLMSettings()
        self.extractor = ExtractorSettings()
        self.claim_graph = ClaimGraphSettings()
        self.clustering = ClusteringSettings()
        self.events = EventSettings()
        self.podcast = PodcastSettings()
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
                    region=src.get("region"),
                    perspective=src.get("perspective"),
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
