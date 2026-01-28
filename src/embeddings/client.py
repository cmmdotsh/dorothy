"""LMStudio embedding API client."""

from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)


class EmbeddingError(Exception):
    """Error generating embeddings."""

    pass


class EmbeddingClient:
    """
    Client for generating embeddings via LMStudio's OpenAI-compatible API.

    LMStudio exposes an OpenAI-compatible /v1/embeddings endpoint.
    """

    def __init__(
        self,
        base_url: str = "http://192.168.0.149:1234",
        model: str = "text-embedding-mxbai-embed-large-v1",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/v1/embeddings"
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)

        Raises:
            EmbeddingError: If the API call fails
        """
        if not texts:
            return []

        try:
            response = self.client.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "input": texts,
                },
            )
            response.raise_for_status()

            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]

            logger.debug("embeddings_generated", count=len(embeddings))
            return embeddings

        except httpx.HTTPStatusError as e:
            logger.error("embedding_api_error", status=e.response.status_code, error=str(e))
            raise EmbeddingError(f"API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error("embedding_request_error", error=str(e))
            raise EmbeddingError(f"Request failed: {e}") from e
        except (KeyError, TypeError) as e:
            logger.error("embedding_parse_error", error=str(e))
            raise EmbeddingError(f"Failed to parse response: {e}") from e

    def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector as a list of floats
        """
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else []

    def health_check(self) -> bool:
        """Check if the embedding service is reachable."""
        try:
            result = self.embed(["test"])
            if result and len(result[0]) > 0:
                logger.info("embedding_service_healthy", dimensions=len(result[0]))
                return True
            return False
        except EmbeddingError:
            return False
