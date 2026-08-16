"""LLM client for story synthesis via Ollama."""

import time
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
    Client for generating text via Ollama's OpenAI-compatible API.

    Used for the primary synthesis pass (article generation + coverage analysis).
    """

    def __init__(
        self,
        base_url: str = "http://192.168.0.149:11434",
        model: str = "qwen3.5:27b",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        context_length: int = 32768,
        timeout: float = 600.0,
        skip_thinking: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_length = context_length
        self.timeout = timeout
        self.skip_thinking = skip_thinking
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
        chat_template_kwargs: Optional[dict] = None,
        skip_thinking: Optional[bool] = None,
    ) -> str:
        """Generate text completion.

        Args:
            skip_thinking: If True, prepend an assistant message with an empty
                <think></think> block to prevent models from entering thinking
                mode. Defaults to the client's configured value when None.
        """
        if skip_thinking is None:
            skip_thinking = self.skip_thinking

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        if skip_thinking:
            messages.append({"role": "assistant", "content": "<think>\n</think>\n"})

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
                        "llm_output_truncated",
                        prompt_length=len(prompt),
                        response_length=len(content),
                        max_tokens=request_json.get("max_tokens"),
                    )

                logger.debug(
                    "llm_generation_complete",
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
                        "llm_retrying",
                        attempt=attempt,
                        max_retries=max_retries,
                        delay=delay,
                        status=e.response.status_code,
                        error_body=body[:200],
                    )
                    time.sleep(delay)
                    continue

                logger.error(
                    "llm_api_error",
                    status=e.response.status_code,
                    error=str(e),
                    response_body=body,
                )
                raise LLMError(f"API error: {e.response.status_code} — {body}") from e

            except httpx.RequestError as e:
                if attempt < max_retries:
                    delay = 10 * attempt
                    logger.warning(
                        "llm_retrying",
                        attempt=attempt,
                        max_retries=max_retries,
                        delay=delay,
                        error=str(e),
                    )
                    time.sleep(delay)
                    continue

                logger.error("llm_request_error", error=str(e))
                raise LLMError(f"Request failed: {e}") from e

            except (KeyError, TypeError, IndexError) as e:
                logger.error("llm_parse_error", error=str(e))
                raise LLMError(f"Failed to parse response: {e}") from e

    def get_max_context_length(self) -> int:
        """Get the configured context length."""
        return self.context_length

    def get_prompt_token_budget(self) -> int:
        """Get the max tokens available for the prompt.

        Reserves space for the completion (max_tokens) plus a safety margin.
        """
        context_length = self.get_max_context_length()
        safety_margin = int(context_length * 0.10)
        return context_length - self.max_tokens - safety_margin

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for a string.

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
