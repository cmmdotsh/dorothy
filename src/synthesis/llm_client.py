"""LLM client for story synthesis via LMStudio."""

from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)


class LLMError(Exception):
    """Error communicating with LLM."""

    pass


class LLMClient:
    """
    Client for generating text via LMStudio's OpenAI-compatible API.

    LMStudio exposes an OpenAI-compatible /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        base_url: str = "http://192.168.0.149:1234",
        model: str = "qwen/qwen3-next-80b",
        temperature: float = 0.3,
        max_tokens: int = 1500,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
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
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Generated text

        Raises:
            LLMError: If the API call fails
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature or self.temperature,
                    "max_tokens": max_tokens or self.max_tokens,
                },
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            logger.debug(
                "llm_generation_complete",
                prompt_length=len(prompt),
                response_length=len(content),
            )

            return content

        except httpx.HTTPStatusError as e:
            logger.error("llm_api_error", status=e.response.status_code, error=str(e))
            raise LLMError(f"API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error("llm_request_error", error=str(e))
            raise LLMError(f"Request failed: {e}") from e
        except (KeyError, TypeError, IndexError) as e:
            logger.error("llm_parse_error", error=str(e))
            raise LLMError(f"Failed to parse response: {e}") from e

    def health_check(self) -> bool:
        """Check if the LLM service is reachable."""
        try:
            result = self.generate("Say 'ok' if you can read this.", max_tokens=10)
            if result:
                logger.info("llm_service_healthy", model=self.model)
                return True
            return False
        except LLMError:
            return False
