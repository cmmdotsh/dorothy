#!/usr/bin/env python3
"""
Dorothy Podcast Generator

Generates NPR-style audio news briefings from synthesized stories.

Usage:
    python -m scripts.run_podcast                  # Full generation (one-shot)
    python -m scripts.run_podcast --script-only    # Just output the script JSON (no TTS)
    python -m scripts.run_podcast --stories 3      # Fewer stories
    python -m scripts.run_podcast --device cuda    # GPU TTS
    python -m scripts.run_podcast --daemon         # Run on schedule (container mode)
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import schedule
import structlog
from rich.console import Console
from rich.panel import Panel

from src.config import config
from src.storage import OpenSearchClient
from src.synthesis.llm_client import LLMClient
from src.podcast.generator import PodcastGenerator

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)
console = Console()


def create_clients() -> tuple[OpenSearchClient, LLMClient]:
    """Create OpenSearch and LLM clients from config."""
    auth_kwargs = {}
    if config.opensearch.username and config.opensearch.password:
        auth_kwargs["username"] = config.opensearch.username
        auth_kwargs["password"] = config.opensearch.password

    os_client = OpenSearchClient(
        host=config.opensearch.host,
        port=config.opensearch.port,
        use_ssl=config.opensearch.use_ssl,
        **auth_kwargs,
    )

    llm_client = LLMClient(
        base_url=config.llm.base_url,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        context_length=config.llm.context_length,
    )

    return os_client, llm_client


def create_generator(
    os_client: OpenSearchClient,
    llm_client: LLMClient,
    args: argparse.Namespace,
) -> PodcastGenerator:
    """Create PodcastGenerator from config and CLI args."""
    return PodcastGenerator(
        os_client=os_client,
        llm_client=llm_client,
        output_dir=Path(args.output),
        tts_device=args.device,
        tts_workers=args.workers,
        voice_ref=config.podcast.voice_ref,
        story_count=args.stories,
        bitrate=config.podcast.bitrate,
        hf_fallback=config.podcast.hf_fallback,
        hf_token=config.podcast.hf_token,
    )


def run_once(args: argparse.Namespace) -> None:
    """Run a single podcast generation."""
    os_client, llm_client = create_clients()

    if not os_client.health_check():
        console.print("[red]OpenSearch unavailable[/red]")
        sys.exit(1)

    generator = create_generator(os_client, llm_client, args)

    if args.script_only:
        console.print("[dim]Generating script only (no TTS)...[/dim]")
        script = generator.generate_script_only()
        if script:
            console.print(Panel.fit("[bold green]Script Generated[/bold green]"))
            console.print_json(json.dumps(script, indent=2))
        else:
            console.print("[red]Script generation failed[/red]")
            sys.exit(1)
    else:
        console.print("[dim]Generating podcast...[/dim]")
        mp3_path = generator.generate()
        if mp3_path:
            console.print(Panel.fit(
                f"[bold green]Podcast Generated[/bold green]\n{mp3_path}"
            ))
        else:
            console.print("[red]Podcast generation failed[/red]")
            sys.exit(1)

    llm_client.close()


def daemon_mode(args: argparse.Namespace) -> None:
    """Run podcast generation on a schedule."""
    console.print(Panel.fit(
        f"[bold green]Dorothy Podcast Daemon[/bold green]\n"
        f"Running every {args.interval} minutes\n"
        f"Press Ctrl+C to stop"
    ))

    os_client, llm_client = create_clients()

    if not os_client.health_check():
        console.print("[red]OpenSearch unavailable[/red]")
        sys.exit(1)

    generator = create_generator(os_client, llm_client, args)

    def run_cycle():
        console.print("\n[dim]Starting podcast generation cycle...[/dim]")
        try:
            mp3_path = generator.generate()
            if mp3_path:
                console.print(f"[green]Podcast generated: {mp3_path}[/green]")
            else:
                console.print("[yellow]Podcast generation skipped or failed[/yellow]")
        except Exception as e:
            logger.error("podcast_cycle_failed", error=str(e))
            console.print(f"[red]Podcast cycle failed: {e}[/red]")

    # Run immediately
    run_cycle()

    # Schedule
    schedule.every(args.interval).minutes.do(run_cycle)

    def shutdown_handler(signum, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
        llm_client.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    while True:
        schedule.run_pending()
        time.sleep(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dorothy Podcast Generator")
    parser.add_argument(
        "--script-only",
        action="store_true",
        help="Generate script JSON only (no TTS/audio)",
    )
    parser.add_argument(
        "--stories",
        type=int,
        default=config.podcast.story_count,
        help=f"Number of stories (default: {config.podcast.story_count})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.podcast.tts_device,
        help=f"TTS device: cpu or cuda (default: {config.podcast.tts_device})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.podcast.tts_workers,
        help=f"Parallel TTS workers (default: {config.podcast.tts_workers})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/podcast",
        help="Output directory (default: output/podcast)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run on schedule (container mode)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Daemon interval in minutes (default: 60)",
    )

    args = parser.parse_args()

    if args.daemon:
        daemon_mode(args)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
