"""Text-to-speech client for Dorothy podcast generation.

Uses mlx-audio with Chatterbox Turbo for MLX-accelerated voice synthesis
on Apple Silicon, with optional HuggingFace Inference API fallback.
"""

from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

MLX_MODEL_ID = "mlx-community/chatterbox-turbo-fp16"


class TTSClient:
    """Wrapper for mlx-audio Chatterbox Turbo with lazy loading and HF fallback."""

    def __init__(
        self,
        hf_fallback: bool = False,
        hf_token: str = "",
        exaggeration: float = 0.3,
        **kwargs,
    ):
        self.hf_fallback = hf_fallback
        self.hf_token = hf_token
        self.exaggeration = exaggeration
        self._model = None
        self._sample_rate: Optional[int] = None

    def is_available(self) -> bool:
        """Check if mlx-audio is importable (or HF fallback is configured)."""
        try:
            import mlx_audio.tts  # noqa: F401

            return True
        except ImportError:
            if self.hf_fallback and self.hf_token:
                return True
            logger.warning(
                "tts_unavailable",
                mlx_audio="not installed",
                hf_fallback=self.hf_fallback,
            )
            return False

    def _load_model(self):
        """Lazy-load the mlx-audio Chatterbox Turbo model."""
        if self._model is not None:
            return

        from mlx_audio.tts.utils import load_model

        logger.info("loading_tts_model", model=MLX_MODEL_ID, backend="mlx")
        self._model = load_model(MLX_MODEL_ID)
        logger.info("tts_model_loaded", model=MLX_MODEL_ID)

    def _synthesize_local(
        self, text: str, output_path: Path, voice_ref_path: Optional[Path] = None
    ) -> Path:
        """Synthesize speech using mlx-audio Chatterbox Turbo."""
        import soundfile as sf
        import numpy as np

        self._load_model()

        voice_ref = voice_ref_path if voice_ref_path and voice_ref_path.exists() else None
        if voice_ref is None and voice_ref_path:
            logger.warning("voice_ref_missing", path=str(voice_ref_path))

        # Generate all chunks and concatenate (stream=False collects into one result)
        results = list(self._model.generate(
            text,
            ref_audio=str(voice_ref) if voice_ref else None,
            exaggeration=self.exaggeration,
            stream=False,
        ))

        if not results:
            raise RuntimeError(f"TTS generated no audio for: {text[:50]}...")

        result = results[0]
        sample_rate = result.sample_rate

        # result.audio is an mlx.core.array — convert to numpy
        audio_np = np.array(result.audio, copy=False)
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        sf.write(str(output_path), audio_np, sample_rate)
        logger.debug(
            "tts_segment_saved",
            path=str(output_path),
            rtf=round(result.real_time_factor, 2),
        )
        return output_path

    def _synthesize_hf(self, text: str, output_path: Path) -> Path:
        """Synthesize speech using HuggingFace Inference API."""
        import httpx

        response = httpx.post(
            "https://router.huggingface.co/hf-inference/models/ResembleAI/chatterbox",
            headers={"Authorization": f"Bearer {self.hf_token}"},
            json={"inputs": text},
            timeout=120.0,
        )
        response.raise_for_status()

        output_path.write_bytes(response.content)
        logger.debug("tts_hf_segment_saved", path=str(output_path))
        return output_path

    def synthesize_to_file(
        self, text: str, output_path: Path, voice_ref_path: Optional[str] = None
    ) -> Path:
        """Generate WAV/FLAC for a text segment.

        Tries local mlx-audio first, falls back to HF API if configured.

        Args:
            text: Text to synthesize.
            output_path: Where to write the audio file.
            voice_ref_path: Optional path to voice reference WAV file.

        Returns:
            Path to the written audio file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path = Path(voice_ref_path) if voice_ref_path else None

        try:
            return self._synthesize_local(text, output_path, voice_ref_path=ref_path)
        except ImportError:
            if self.hf_fallback and self.hf_token:
                logger.info("tts_falling_back_to_hf")
                return self._synthesize_hf(text, output_path)
            raise
