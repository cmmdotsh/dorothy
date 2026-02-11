"""LLM client for story synthesis via LMStudio."""

from typing import Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)


class LLMError(Exception):
    """Error communicating with LLM."""

    pass


# Conservative chars-per-token ratio for estimation.
# English text averages ~4 chars/token; we use 3.5 for safety margin.
CHARS_PER_TOKEN = 3.5


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
        timeout: float = 600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None
        self._max_context_length: Optional[int] = None

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
            # Log the response body — LMStudio includes useful error details
            body = ""
            try:
                body = e.response.text[:500]
            except Exception:
                pass
            logger.error(
                "llm_api_error",
                status=e.response.status_code,
                error=str(e),
                response_body=body,
            )
            raise LLMError(f"API error: {e.response.status_code} — {body}") from e
        except httpx.RequestError as e:
            logger.error("llm_request_error", error=str(e))
            raise LLMError(f"Request failed: {e}") from e
        except (KeyError, TypeError, IndexError) as e:
            logger.error("llm_parse_error", error=str(e))
            raise LLMError(f"Failed to parse response: {e}") from e

    def get_max_context_length(self) -> int:
        """
        Get the model's actual loaded context length from LMStudio.

        LMStudio's model info endpoint returns `max_context_length` which is the
        theoretical maximum, NOT the actual loaded context. The real loaded context
        is returned in `model_info.context_length` inside completion responses.

        We do a tiny completion to discover the true loaded value.
        Falls back to the model info endpoint, then a conservative default.
        """
        if self._max_context_length is not None:
            return self._max_context_length

        # Strategy 1: Do a tiny completion and read model_info.context_length
        try:
            response = self.client.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
            )
            response.raise_for_status()
            data = response.json()
            model_info = data.get("model_info", {})
            loaded_ctx = model_info.get("context_length")

            if loaded_ctx and loaded_ctx > 0:
                self._max_context_length = loaded_ctx
                logger.info(
                    "model_context_length",
                    model=self.model,
                    loaded_context_length=self._max_context_length,
                    source="completion_model_info",
                )
                return self._max_context_length
        except Exception as e:
            logger.debug("context_probe_failed", error=str(e))

        # Strategy 2: Fall back to model info endpoint (may be theoretical max)
        try:
            response = self.client.get(f"{self.base_url}/api/v0/models/{self.model}")
            response.raise_for_status()
            data = response.json()
            self._max_context_length = data.get("max_context_length", 32768)
            logger.warning(
                "using_theoretical_context_length",
                model=self.model,
                max_context_length=self._max_context_length,
                msg="Could not determine actual loaded context; using model max which may be too large",
            )
            return self._max_context_length
        except Exception as e:
            logger.warning(
                "failed_to_get_context_length",
                error=str(e),
                fallback=32768,
            )
            self._max_context_length = 32768
            return self._max_context_length

    def get_prompt_token_budget(self) -> int:
        """
        Get the max tokens available for the prompt.

        Reserves space for the completion (max_tokens) plus a safety margin.
        """
        context_length = self.get_max_context_length()
        # Reserve max_tokens for completion + 10% safety margin
        safety_margin = int(context_length * 0.10)
        return context_length - self.max_tokens - safety_margin

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count for a string.

        Uses a conservative chars-per-token ratio. Not exact, but sufficient
        for budget decisions ("does this fit or do we need to sample?").
        """
        return int(len(text) / CHARS_PER_TOKEN)

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
