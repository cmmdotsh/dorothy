"""Audio assembly for Dorothy podcast — concatenates WAV segments into MP3."""

from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class AudioAssembler:
    """Concatenates audio segments with silence gaps and exports as MP3."""

    def __init__(self, bitrate: str = "128k", atempo: float = 1.0):
        self.bitrate = bitrate
        self.atempo = atempo

    @staticmethod
    def _section_label(path: Path) -> str:
        """Extract section label from filename like '003-story2.wav' → 'story2'."""
        name = path.stem
        # Strip leading numeric prefix (e.g. "003-")
        parts = name.split("-", 1)
        return parts[1] if len(parts) > 1 else name

    def assemble(
        self,
        segment_paths: list[Path],
        output_path: Path,
        chunk_gap_ms: int = 150,
        section_gap_ms: int = 800,
        lead_silence_ms: int = 500,
        trail_silence_ms: int = 1000,
        crossfade_ms: int = 80,
    ) -> Path:
        """Concatenate audio segments with silence gaps and export as MP3.

        Uses short gaps between chunks within the same section (e.g. two
        chunks of the same story) and longer gaps between sections (e.g.
        between stories, or between intro and first story). A crossfade is
        applied at each join to smooth out abrupt transitions.

        Args:
            segment_paths: Ordered list of WAV/FLAC file paths.
            output_path: Where to write the final audio file.
            chunk_gap_ms: Silence between chunks of the same section.
            section_gap_ms: Silence between different sections.
            lead_silence_ms: Silence before first segment.
            trail_silence_ms: Silence after last segment.
            crossfade_ms: Crossfade duration at each segment join.

        Returns:
            Path to the output file.
        """
        from pydub import AudioSegment

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        combined = AudioSegment.silent(duration=lead_silence_ms)
        chunk_gap = AudioSegment.silent(duration=chunk_gap_ms)
        section_gap = AudioSegment.silent(duration=section_gap_ms)

        prev_section = None
        for i, path in enumerate(segment_paths):
            if not path.exists():
                logger.warning("segment_missing", path=str(path), index=i)
                continue

            segment = AudioSegment.from_file(str(path))
            section = self._section_label(path)

            if i > 0:
                if section != prev_section:
                    combined += section_gap
                else:
                    combined += chunk_gap

            # Crossfade into the segment to smooth abrupt transitions.
            # Clamp to avoid crossfading more than the segment length.
            fade = min(crossfade_ms, len(segment) // 2, len(combined) // 2)
            if fade > 0:
                combined = combined.append(segment, crossfade=fade)
            else:
                combined += segment
            prev_section = section

        combined += AudioSegment.silent(duration=trail_silence_ms)

        # Force mono for speech
        combined = combined.set_channels(1)

        # Normalize to -1 dBFS to ensure consistent volume without clipping
        target_dbfs = -1.0
        gain_needed = target_dbfs - combined.max_dBFS
        if gain_needed != 0:
            combined = combined.apply_gain(gain_needed)
            logger.info("audio_normalized", gain_db=round(gain_needed, 1), target_dbfs=target_dbfs)

        # Apply atempo time-stretch via ffmpeg (pitch-preserving speed change)
        ffmpeg_params = []
        if self.atempo != 1.0:
            ffmpeg_params = ["-filter:a", f"atempo={self.atempo}"]
            logger.info("audio_atempo", speed=self.atempo)

        output_format = output_path.suffix.lstrip(".")
        if output_format == "mp3":
            combined.export(
                str(output_path), format="mp3", bitrate=self.bitrate,
                parameters=ffmpeg_params or None,
            )
        else:
            combined.export(
                str(output_path), format=output_format,
                parameters=ffmpeg_params or None,
            )

        duration_secs = len(combined) / 1000.0
        logger.info(
            "audio_assembled",
            segments=len(segment_paths),
            duration_seconds=round(duration_secs, 1),
            output=str(output_path),
        )

        return output_path
