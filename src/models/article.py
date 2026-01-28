"""Article and Source data models for Dorothy news aggregation."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class BiasRating(str, Enum):
    """News source political bias rating (AllSides scale)."""

    LEFT = "left"
    LEAN_LEFT = "lean-left"
    CENTER = "center"
    LEAN_RIGHT = "lean-right"
    RIGHT = "right"


class Column(str, Enum):
    """News column/category classification."""

    POLITICS = "politics"
    LOCAL = "local"
    SPORTS = "sports"
    MONEY = "money"
    LIFESTYLE = "lifestyle"
    TECH = "tech"


class FetchMethod(str, Enum):
    """Method used to fetch articles."""

    RSS = "rss"
    SCRAPE = "scrape"  # Deferred to later phase


class Article(BaseModel):
    """
    Canonical article model for storage and processing.

    This is the normalized format all articles are converted to,
    regardless of source or fetch method.
    """

    id: UUID = Field(default_factory=uuid4)

    # Source metadata
    source_name: str = Field(..., description="Human-readable source name")
    source_slug: str = Field(..., description="URL-safe source identifier")
    source_bias: BiasRating
    column: Column

    # Article content
    headline: str = Field(..., min_length=1, max_length=500)
    summary: Optional[str] = Field(None, max_length=2000)
    url: HttpUrl

    # Timestamps
    pub_date: datetime = Field(..., description="When the article was published")
    fetched_at: datetime = Field(
        default_factory=_utcnow, description="When we fetched the article"
    )

    # Future: embedding vector (Phase 2)
    embedding: Optional[list[float]] = Field(
        None, description="Semantic embedding vector for clustering"
    )

    # Image from RSS feed
    image_url: Optional[str] = Field(
        None, description="Article thumbnail/hero image URL"
    )

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    }

    def to_opensearch_doc(self) -> dict:
        """Convert to OpenSearch document format."""
        return {
            "id": str(self.id),
            "source_name": self.source_name,
            "source_slug": self.source_slug,
            "source_bias": self.source_bias.value,
            "column": self.column.value,
            "headline": self.headline,
            "summary": self.summary,
            "url": str(self.url),
            "pub_date": self.pub_date.isoformat(),
            "fetched_at": self.fetched_at.isoformat(),
            "embedding": self.embedding,
            "image_url": self.image_url,
        }


class Source(BaseModel):
    """News source configuration model."""

    name: str
    slug: str
    rss_url: Optional[HttpUrl] = None
    fetch_method: FetchMethod
    column: Column
    bias: BiasRating
    active: bool = True
