"""LLM client for story synthesis via LMStudio."""

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
    Client for generating text via LMStudio's OpenAI-compatible API.

    Manages model lifecycle: ensures the model is loaded with the correct
    context length before synthesis, and prevents auto-unload via ttl=-1.
    """

    def __init__(
        self,
        base_url: str = "http://192.168.0.149:1234",
        model: str = "qwen/qwen3-next-80b",
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
        self._max_context_length: Optional[int] = None
        self._model_verified = False

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

    def _get_loaded_model_info(self) -> Optional[dict]:
        """Check if our model is currently loaded and return its config."""
        try:
            response = self.client.get(f"{self.base_url}/api/v1/models")
            response.raise_for_status()
            data = response.json()

            for model in data.get("models", []):
                if model.get("key") == self.model:
                    instances = model.get("loaded_instances", [])
                    if instances:
                        return instances[0]
            return None
        except Exception as e:
            logger.debug("model_info_check_failed", error=str(e))
            return None

    def _load_model(self) -> bool:
        """Load the model via LMStudio API with our desired context length."""
        try:
            logger.info(
                "loading_model",
                model=self.model,
                context_length=self.context_length,
            )
            response = self.client.post(
                f"{self.base_url}/api/v1/models/load",
                json={
                    "model": self.model,
                    "context_length": self.context_length,
                    "echo_load_config": True,
                },
            )
            response.raise_for_status()
            data = response.json()

            load_config = data.get("load_config", {})
            loaded_ctx = load_config.get("context_length", 0)
            load_time = data.get("load_time_seconds", 0)

            logger.info(
                "model_loaded",
                model=self.model,
                context_length=loaded_ctx,
                load_time_seconds=round(load_time, 1),
            )

            self._max_context_length = loaded_ctx
            return True
        except Exception as e:
            logger.error("model_load_failed", model=self.model, error=str(e))
            return False

    def _unload_model(self) -> bool:
        """Unload the model via LMStudio API."""
        try:
            response = self.client.post(
                f"{self.base_url}/api/v1/models/unload",
                json={"instance_id": self.model},
            )
            response.raise_for_status()
            logger.info("model_unloaded", model=self.model)
            return True
        except Exception as e:
            logger.error("model_unload_failed", model=self.model, error=str(e))
            return False

    def ensure_model_loaded(self) -> bool:
        """Ensure the model is loaded with the correct context length.

        Checks current state and loads/reloads as needed.
        """
        instance = self._get_loaded_model_info()

        if instance:
            loaded_ctx = instance.get("config", {}).get("context_length", 0)

            if loaded_ctx == self.context_length:
                if not self._model_verified:
                    logger.info(
                        "model_already_loaded",
                        model=self.model,
                        context_length=loaded_ctx,
                    )
                    self._model_verified = True
                self._max_context_length = loaded_ctx
                return True

            # Loaded but with wrong context — reload
            logger.warning(
                "model_context_mismatch",
                model=self.model,
                loaded=loaded_ctx,
                desired=self.context_length,
            )
            self._unload_model()

        # Not loaded — load it
        self._model_verified = False
        return self._load_model()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """
        Generate text completion.

        Ensures model is loaded before generating. Passes ttl=-1 to
        prevent LMStudio from auto-unloading the model between requests.
        """
        # Ensure model is loaded with correct settings
        if not self._model_verified:
            self.ensure_model_loaded()

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
            "ttl": -1,
        }
        if response_format is not None:
            request_json["response_format"] = response_format

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.post(self.endpoint, json=request_json)
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
                body = ""
                try:
                    body = e.response.text[:500]
                except Exception:
                    pass

                # Model unloaded — reload and retry immediately
                if "Model unloaded" in body or "context length" in body.lower():
                    logger.warning("model_lost_mid_session", error=body)
                    self._model_verified = False
                    self.ensure_model_loaded()

                # Transient model crash — wait and retry
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
                    self._model_verified = False
                    self.ensure_model_loaded()
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
        """Get the model's loaded context length.

        Uses the known context_length from our load config, since we
        control the model lifecycle via ensure_model_loaded().
        """
        if self._max_context_length is not None:
            return self._max_context_length

        # Make sure model is loaded so we know the actual context
        self.ensure_model_loaded()
        if self._max_context_length:
            return self._max_context_length

        # Fallback
        self._max_context_length = self.context_length
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
        """Check if the LLM service is reachable and model loads correctly."""
        try:
            if not self.ensure_model_loaded():
                logger.error("health_check_model_load_failed", model=self.model)
                return False

            result = self.generate("Say 'ok' if you can read this.", max_tokens=10)
            if result:
                logger.info("llm_service_healthy", model=self.model)
                return True
            return False
        except LLMError:
            return False
