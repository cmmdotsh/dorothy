# Dorothy — Podcast

Dorothy generates an NPR-style audio news briefing from the day's synthesized stories. The podcast pipeline runs independently from the main synthesis pipeline and is optional.

## Overview

```
Synthesized stories (OpenSearch)
        │
        ▼
  ScriptWriter       LLM → JSON script (intro, stories, outro)
        │
        ▼
  TTSClient          Chatterbox Turbo (mlx-audio) → WAV segments per sentence chunk
        │
        ▼
  AudioAssembler     Concatenate WAVs + silence gaps + crossfade → MP3 (FFmpeg)
        │
        ▼
  FeedGenerator      Update feed.xml (RSS 2.0 podcast feed)
        │
        ▼
  S3Deployer         Upload MP3 + feed.xml + manifest to S3
```

## Running the podcast generator

### One-shot

```bash
python -m scripts.run_podcast
```

### With options

```bash
# Generate 3 stories using CUDA GPU
python -m scripts.run_podcast --stories 3 --device cuda

# Generate script JSON only (no TTS, good for testing)
python -m scripts.run_podcast --script-only

# Generate and deploy to S3
python -m scripts.run_podcast --publish
```

### Daemon mode

```bash
python -m scripts.run_podcast --daemon --interval 60 --publish
```

| Flag | Default | Description |
|------|---------|-------------|
| `--stories N` | `5` (from `PODCAST_STORY_COUNT`) | Stories per episode |
| `--device` | `cpu` (from `PODCAST_TTS_DEVICE`) | PyTorch device: `cpu`, `cuda`, `mps` |
| `--workers N` | `1` (from `PODCAST_TTS_WORKERS`) | Parallel TTS threads |
| `--output PATH` | `output/podcast` | Output directory |
| `--script-only` | off | Output script JSON without generating audio |
| `--daemon` | off | Run continuously on a schedule |
| `--interval N` | `60` | Minutes between episodes (daemon mode) |
| `--publish` | off | Deploy to S3 after each episode |

In Docker, the podcast container runs as a daemon:

```bash
docker-compose up podcast
```

This runs:
```
python -m scripts.run_podcast --daemon --interval 60 --publish
```

## Voice configuration

The podcast uses two voice anchors (A and B) that alternate per segment. Both are configured via reference WAV files for voice cloning:

```
config/voices/
├── anchor_a.wav    # Voice reference for anchor A (PODCAST_VOICE_REF_A)
└── anchor_b.wav    # Voice reference for anchor B (PODCAST_VOICE_REF_B)
```

If `anchor_b.wav` does not exist, the generator falls back to using anchor A's voice for both roles ("single-anchor mode"). The reference WAVs should be clear recordings of 5–30 seconds.

## Pipeline stages

### 1. Story selection

`PodcastGenerator.generate()` selects stories for the episode:

1. Fetch current synthesized stories from the `politics` column (currently the only column used by the podcast)
2. Load recent episode manifests from `output/podcast/*.manifest.json` to identify recently-covered stories
3. Apply recency penalties to hotness scores:
   - Stories in the most recent episode: `hotness × 0.15`
   - Stories in 2–3 episodes ago: `hotness × 0.40`
   - Stories with 3+ new article URLs since last coverage: penalty waived
4. Sort by penalized hotness score
5. Take the top `story_count` stories

### 2. Script writing

`ScriptWriter.write_script(stories)` generates a radio script via LLM:

- System prompt: NPR-style news anchor writing broadcast scripts
- Input: Synthesized article text and metadata for each story
- Constraints enforced by the prompt:
  - 120–150 words per story (≈ 50–60 seconds of audio)
  - Write for the ear: no abbreviations, symbols, or complex clauses
  - Spell everything out: "percent" not "%", "United States" not "US", etc.
  - Story body must not repeat the headline
  - Natural transitions between stories
- Output JSON:
  ```json
  {
    "intro": "From Dorothy, it's Friday, March 13th...",
    "stories": [
      {
        "headline_read": "Spoken one-sentence headline",
        "body": "120-150 word broadcast script for this story"
      }
    ],
    "outro": "That's your Dorothy news update..."
  }
  ```

Each story in the prompt is truncated to 300 words to stay within the LLM's context budget.

### 3. Text-to-speech

`TTSClient` converts script text to audio using Chatterbox Turbo (mlx-audio):

- Primary engine: `mlx-audio` Chatterbox Turbo (MLX-accelerated, best on Apple Silicon)
- Optional fallback: HuggingFace Inference API (set `PODCAST_HF_FALLBACK=true`)
- Voice cloning via the reference WAV files

Text is split into 2-sentence chunks before synthesis. Chatterbox degrades on long inputs, so shorter chunks produce better audio quality.

Anchors alternate:
- Intro: anchor A
- Story 1: anchor A
- Story 2: anchor B
- Story 3: anchor A
- ...
- Outro: anchor A

Each chunk is synthesized to a temporary WAV file.

### 4. Audio assembly

`AudioAssembler.assemble(segment_wavs, output_path)` combines all WAV files into a final MP3:

**Timing:**
- Lead silence: 500ms before first segment
- Within-story gap: 150ms between sentence chunks of the same story
- Between-story gap: 800ms between stories / intro / outro
- Trail silence: 1000ms at the end

**Post-processing:**
- 80ms crossfade at each join to smooth transitions
- `atempo` filter (default 1.1× — 10% speed-up for natural broadcast pace)
- Export to MP3 at configured bitrate (default 128k) via FFmpeg/pydub

### 5. Feed generation

`generate_feed(output_dir, base_url)` creates or updates `feed.xml`:

- RSS 2.0 format with iTunes podcast extensions
- Episode title: `"Dorothy News Briefing — {date}"`
- Channel metadata: author, category (`News`), explicit (`no`)
- Enclosure: MP3 URL, file size in bytes, duration
- Rolling window: keeps the 24 most recent episodes
- `latest.mp3` symlink/copy is also maintained for direct playback

### 6. Deployment

Only changed files are uploaded to S3:
- The new episode's MP3
- Its manifest JSON
- `feed.xml` (always)
- `latest.mp3` (always)

Previous episodes already on S3 are not re-uploaded.

## Output directory structure

```
output/podcast/
├── feed.xml                              # RSS podcast feed
├── latest.mp3                            # Copy of the most recent episode
├── dorothy-2026-03-13T14-00-00.mp3       # Episode audio
├── dorothy-2026-03-13T14-00-00.manifest.json  # Episode metadata
├── dorothy-2026-03-12T14-00-00.mp3
├── dorothy-2026-03-12T14-00-00.manifest.json
└── .tmp/                                 # Temporary WAV segments (cleaned up after assembly)
```

**Manifest JSON:**
```json
{
  "episode_id": "dorothy-2026-03-13T14-00-00",
  "generated_at": "2026-03-13T14:05:23Z",
  "story_ids": ["abc123", "def456", "ghi789", "jkl012", "mno345"],
  "article_urls": ["https://...", "https://..."],
  "story_count": 5
}
```

The manifest is used by the next episode to apply recency penalties.

## Dependencies

The podcast feature requires additional dependencies not installed by default:

```bash
pip install ".[podcast]"
# or
pip install chatterbox-tts pydub torchaudio
```

FFmpeg must also be installed and available on `PATH` (used by pydub for MP3 encoding).

The podcast Docker image (`Dockerfile.podcast`) handles all of this automatically.

## Hardware notes

| Device | Performance | Notes |
|--------|-------------|-------|
| `mps` | Fast | Apple Silicon (M1/M2/M3); mlx-audio is optimized for this |
| `cuda` | Fast | NVIDIA GPU |
| `cpu` | Slow | Falls back gracefully; a 5-story episode takes several minutes |

For Apple Silicon hosts, use the mlx-audio path (default) with `--device mps` or leave it as `cpu` and let mlx-audio use the Metal backend automatically.
