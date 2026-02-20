"""Text-to-speech client for Dorothy podcast generation.

Uses Chatterbox-TTS for local voice synthesis, with optional HuggingFace
Inference API fallback.
"""

from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class TTSClient:
    """Wrapper for Chatterbox-TTS with lazy loading and HF fallback."""

    def __init__(
        self,
        voice_ref_path: str = "config/voices/default.wav",
        device: str = "cpu",
        hf_fallback: bool = False,
        hf_token: str = "",
        exaggeration: float = 0.3,
    ):
        self.voice_ref_path = Path(voice_ref_path)
        self.device = device
        self.hf_fallback = hf_fallback
        self.hf_token = hf_token
        self.exaggeration = exaggeration
        self._model = None
        self._sample_rate: Optional[int] = None

    def is_available(self) -> bool:
        """Check if Chatterbox-TTS is importable (or HF fallback is configured)."""
        try:
            import chatterbox.tts  # noqa: F401

            return True
        except ImportError:
            if self.hf_fallback and self.hf_token:
                return True
            logger.warning(
                "tts_unavailable",
                chatterbox="not installed",
                hf_fallback=self.hf_fallback,
            )
            return False

    def _load_model(self):
        """Lazy-load the Chatterbox model."""
        if self._model is not None:
            return

        import torch
        from chatterbox.tts import ChatterboxTTS

        logger.info("loading_tts_model", device=self.device)
        self._model = ChatterboxTTS.from_pretrained(device=self.device)
        self._sample_rate = self._model.sr
        logger.info("tts_model_loaded", sample_rate=self._sample_rate)

    def _synthesize_local(self, text: str, output_path: Path) -> Path:
        """Synthesize speech using local Chatterbox model."""
        import soundfile as sf

        self._load_model()

        voice_ref = self.voice_ref_path if self.voice_ref_path.exists() else None
        if voice_ref is None:
            logger.warning("voice_ref_missing", path=str(self.voice_ref_path))

        wav = self._model.generate(
            text,
            audio_prompt_path=str(voice_ref) if voice_ref else None,
            exaggeration=self.exaggeration,
        )

        # wav is a torch tensor [1, samples] — convert to numpy for soundfile
        audio_np = wav.squeeze(0).cpu().numpy()
        sf.write(str(output_path), audio_np, self._sample_rate)
        logger.debug("tts_segment_saved", path=str(output_path))
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

    def synthesize_to_file(self, text: str, output_path: Path) -> Path:
        """Generate WAV/FLAC for a text segment.

        Tries local Chatterbox first, falls back to HF API if configured.

        Args:
            text: Text to synthesize.
            output_path: Where to write the audio file.

        Returns:
            Path to the written audio file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            return self._synthesize_local(text, output_path)
        except ImportError:
            if self.hf_fallback and self.hf_token:
                logger.info("tts_falling_back_to_hf")
                return self._synthesize_hf(text, output_path)
            raise
