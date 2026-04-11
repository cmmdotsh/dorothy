"""Ollama inference client for story synthesis and article review."""

import time
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)


class OllamaError(Exception):
    """Error communicating with Ollama."""

    pass


# Conservative chars-per-token ratio for estimation.
CHARS_PER_TOKEN = 3.5


class OllamaClient:
    """
    Client for generating text via Ollama's OpenAI-compatible API.

    Unlike LMStudio, Ollama manages model lifecycle automatically —
    no explicit load/unload needed. Models are loaded on first request
    and kept in memory based on OLLAMA_KEEP_ALIVE.
    """

    def __init__(
        self,
        base_url: str = "http://192.168.0.149:11434",
        model: str = "gemma4:31b",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        context_length: int = 32768,
        timeout: float = 600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_length = context_length
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

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """Generate text completion via Ollama's OpenAI-compatible API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        max_retries = 3
        request_json = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if response_format is not None:
            request_json["response_format"] = response_format

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.post(self.endpoint, json=request_json)
                response.raise_for_status()

                data = response.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "unknown")

                if finish_reason == "length":
                    logger.warning(
                        "ollama_output_truncated",
                        prompt_length=len(prompt),
                        response_length=len(content),
                        max_tokens=request_json.get("max_tokens"),
                    )

                logger.debug(
                    "ollama_generation_complete",
                    model=self.model,
                    prompt_length=len(prompt),
                    response_length=len(content),
                    finish_reason=finish_reason,
                )

                return content

            except httpx.HTTPStatusError as e:
                body = ""
                try:
                    body = e.response.text[:500]
                except Exception:
                    pass

                if attempt < max_retries:
                    delay = 10 * attempt
                    logger.warning(
                        "ollama_retrying",
                        attempt=attempt,
                        max_retries=max_retries,
                        delay=delay,
                        status=e.response.status_code,
                        error_body=body[:200],
                    )
                    time.sleep(delay)
                    continue

                logger.error(
                    "ollama_api_error",
                    status=e.response.status_code,
                    error=str(e),
                    response_body=body,
                )
                raise OllamaError(f"API error: {e.response.status_code} — {body}") from e

            except httpx.RequestError as e:
                if attempt < max_retries:
                    delay = 10 * attempt
                    logger.warning(
                        "ollama_retrying",
                        attempt=attempt,
                        max_retries=max_retries,
                        delay=delay,
                        error=str(e),
                    )
                    time.sleep(delay)
                    continue

                logger.error("ollama_request_error", error=str(e))
                raise OllamaError(f"Request failed: {e}") from e

            except (KeyError, TypeError, IndexError) as e:
                logger.error("ollama_parse_error", error=str(e))
                raise OllamaError(f"Failed to parse response: {e}") from e

    def get_prompt_token_budget(self) -> int:
        """Get the max tokens available for the prompt.

        Reserves space for the completion (max_tokens) plus a safety margin.
        """
        safety_margin = int(self.context_length * 0.10)
        return self.context_length - self.max_tokens - safety_margin

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for a string."""
        return int(len(text) / CHARS_PER_TOKEN)

    def health_check(self) -> bool:
        """Check if Ollama is reachable and has models available."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            logger.info(
                "ollama_healthy",
                base_url=self.base_url,
                models_available=len(models),
            )
            return True
        except Exception as e:
            logger.error("ollama_health_check_failed", error=str(e))
            return False
