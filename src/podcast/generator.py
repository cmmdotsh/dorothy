"""Podcast generation orchestrator for Dorothy."""

import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Optional

import structlog

from src.config import config
from src.podcast.script_writer import ScriptWriter
from src.podcast.feed import generate_feed
from src.storage import OpenSearchClient
from src.synthesis.llm_client import LLMClient, LLMError

logger = structlog.get_logger(__name__)

# Regex to split on sentence boundaries (period, !, ? followed by space or end)
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def _chunk_sentences(text: str, sentences_per_chunk: int = 2) -> list[str]:
    """Split text into chunks of N sentences each.

    Chatterbox TTS produces better audio with shorter inputs — long text
    leads to quality degradation and eventual cutoff.
    """
    sentences = _SENTENCE_RE.split(text.strip())
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk)
    return chunks

COLUMNS = ["politics"]


class PodcastGenerator:
    """Orchestrates the full podcast generation pipeline:
    story selection → script writing → TTS → audio assembly → feed update.
    """

    def __init__(
        self,
        os_client: OpenSearchClient,
        llm_client: LLMClient,
        output_dir: Path = Path("output/podcast"),
        tts_device: str = "cpu",
        tts_workers: int = 1,
        voice_ref: str = "config/voices/default.wav",
        story_count: int = 5,
        bitrate: str = "128k",
        hf_fallback: bool = False,
        hf_token: str = "",
    ):
        self.os_client = os_client
        self.llm_client = llm_client
        self.output_dir = output_dir
        self.tmp_dir = output_dir / ".tmp"
        self.tts_device = tts_device
        self.tts_workers = max(1, tts_workers)
        self.voice_ref = voice_ref
        self.story_count = story_count
        self.bitrate = bitrate
        self.hf_fallback = hf_fallback
        self.hf_token = hf_token

        self.script_writer = ScriptWriter(llm_client)

    def _get_tts_client(self):
        """Lazy-import and create TTS client."""
        from src.podcast.tts_client import TTSClient

        return TTSClient(
            voice_ref_path=self.voice_ref,
            device=self.tts_device,
            hf_fallback=self.hf_fallback,
            hf_token=self.hf_token,
        )

    def _get_assembler(self):
        """Lazy-import and create audio assembler."""
        from src.podcast.audio_assembler import AudioAssembler

        return AudioAssembler(bitrate=self.bitrate)

    def _select_stories(self) -> list[dict]:
        """Pull top stories from OpenSearch across all columns."""
        syntheses_by_column = {}
        for column in COLUMNS:
            stories = self.os_client.get_syntheses(column=column, limit=3)
            if stories:
                syntheses_by_column[column] = stories

        if not syntheses_by_column:
            logger.warning("no_syntheses_available")
            return []

        return self.script_writer.select_top_stories(
            syntheses_by_column, count=self.story_count
        )

    def generate_script_only(self) -> Optional[dict]:
        """Generate just the radio script (no TTS). Useful for testing.

        Returns:
            Parsed script dict, or None on failure.
        """
        stories = self._select_stories()
        if len(stories) < 2:
            logger.warning("too_few_stories", count=len(stories))
            return None

        now = datetime.now(timezone.utc)
        eastern = now.astimezone(ZoneInfo("America/New_York"))
        hour = eastern.strftime("%I").lstrip("0")
        am_pm = eastern.strftime("%p").upper()
        dateline = f"{eastern.strftime('%B')} {eastern.day}, {eastern.year}, {hour} {am_pm} Eastern"

        try:
            return self.script_writer.generate_script(stories, dateline=dateline)
        except (LLMError, Exception) as e:
            logger.error("script_generation_failed", error=str(e))
            return None

    def generate(self) -> Optional[Path]:
        """Run the full podcast generation pipeline.

        Returns:
            Path to the generated MP3, or None on failure.
        """
        now = datetime.now(timezone.utc)
        eastern = now.astimezone(ZoneInfo("America/New_York"))
        hour = eastern.strftime("%I").lstrip("0")
        am_pm = eastern.strftime("%p").upper()
        dateline = f"{eastern.strftime('%B')} {eastern.day}, {eastern.year}, {hour} {am_pm} Eastern"
        timestamp = now.strftime("%Y%m%d-%H%M")

        # 1. Select stories
        stories = self._select_stories()
        if len(stories) < 2:
            logger.warning("too_few_stories_for_podcast", count=len(stories))
            return None

        logger.info("podcast_stories_selected", count=len(stories))

        # 2. Generate script
        try:
            script = self.script_writer.generate_script(stories, dateline=dateline)
        except (LLMError, Exception) as e:
            logger.error("podcast_script_failed", error=str(e))
            return None

        # 3. TTS each segment
        try:
            tts = self._get_tts_client()
        except ImportError:
            logger.error("tts_not_available", hint="Install chatterbox-tts")
            return None

        if not tts.is_available():
            logger.error("tts_not_available")
            return None

        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compute all chunks with their ordering index
        all_chunks: list[tuple[int, str, str]] = []  # (index, text, label)
        chunk_idx = 0

        for chunk_text in _chunk_sentences(script["intro"], sentences_per_chunk=3):
            all_chunks.append((chunk_idx, chunk_text, "intro"))
            chunk_idx += 1

        for i, story in enumerate(script["stories"]):
            full_text = f"{story['headline_read']} {story['body']}"
            for chunk_text in _chunk_sentences(full_text, sentences_per_chunk=3):
                all_chunks.append((chunk_idx, chunk_text, f"story{i + 1}"))
                chunk_idx += 1

        for chunk_text in _chunk_sentences(script["outro"], sentences_per_chunk=3):
            all_chunks.append((chunk_idx, chunk_text, "outro"))
            chunk_idx += 1

        logger.info(
            "tts_chunks_prepared",
            total=len(all_chunks),
            workers=self.tts_workers,
        )

        # Synthesize chunks (parallel when workers > 1)
        segment_paths: list[Optional[Path]] = [None] * len(all_chunks)

        def _synth_one(idx: int, text: str, label: str) -> tuple[int, Optional[Path]]:
            wav_path = self.tmp_dir / f"{idx:03d}-{label}.wav"
            try:
                tts.synthesize_to_file(text, wav_path)
                return (idx, wav_path)
            except Exception as e:
                logger.warning("tts_chunk_failed", label=label, idx=idx, error=str(e))
                return (idx, None)

        if self.tts_workers <= 1:
            for idx, text, label in all_chunks:
                i, path = _synth_one(idx, text, label)
                segment_paths[i] = path
        else:
            with ThreadPoolExecutor(max_workers=self.tts_workers) as pool:
                futures = {
                    pool.submit(_synth_one, idx, text, label): idx
                    for idx, text, label in all_chunks
                }
                for future in as_completed(futures):
                    i, path = future.result()
                    segment_paths[i] = path

        # Filter out failed chunks, preserving order
        segment_paths = [p for p in segment_paths if p is not None]

        if len(segment_paths) < 2:
            logger.error("too_few_segments", count=len(segment_paths))
            self._cleanup_tmp()
            return None

        # 4. Assemble
        try:
            assembler = self._get_assembler()
        except ImportError:
            logger.error("pydub_not_available", hint="Install pydub")
            self._cleanup_tmp()
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        mp3_path = self.output_dir / f"dorothy-{timestamp}.mp3"

        try:
            assembler.assemble(segment_paths, mp3_path)
        except Exception as e:
            logger.error("audio_assembly_failed", error=str(e))
            self._cleanup_tmp()
            return None

        # 5. Copy to latest.mp3
        latest_path = self.output_dir / "latest.mp3"
        shutil.copy2(mp3_path, latest_path)

        # 6. Update RSS feed
        try:
            generate_feed(self.output_dir)
        except Exception as e:
            logger.warning("feed_generation_failed", error=str(e))

        # 7. Cleanup temp
        self._cleanup_tmp()

        logger.info(
            "podcast_generated",
            path=str(mp3_path),
            segments=len(segment_paths),
        )

        return mp3_path

    def _cleanup_tmp(self):
        """Remove temporary WAV files."""
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
