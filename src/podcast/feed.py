"""RSS podcast feed generator for Dorothy news briefings."""

import os
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent

import structlog

logger = structlog.get_logger(__name__)

SITE_URL = "https://dorothy.cmm.sh"
PODCAST_URL = f"{SITE_URL}/podcast"


def generate_feed(
    podcast_dir: Path,
    max_episodes: int = 24,
    output_path: Path | None = None,
) -> Path:
    """Generate an RSS 2.0 podcast feed from MP3 files in podcast_dir.

    Scans for dorothy-*.mp3 files, sorts by name (newest first), and
    writes a podcast-compatible RSS feed.

    Args:
        podcast_dir: Directory containing MP3 episode files.
        max_episodes: Maximum episodes to include (rolling window).
        output_path: Where to write feed.xml. Defaults to podcast_dir/feed.xml.

    Returns:
        Path to the written feed.xml.
    """
    if output_path is None:
        output_path = podcast_dir / "feed.xml"

    # Find episode MP3s (named dorothy-YYYYMMDD-HHMM.mp3)
    episodes = sorted(podcast_dir.glob("dorothy-*.mp3"), reverse=True)[:max_episodes]

    rss = Element("rss", version="2.0")
    rss.set("xmlns:itunes", "http://www.itunes.com/dtds/podcast-1.0.dtd")

    channel = SubElement(rss, "channel")
    SubElement(channel, "title").text = "Dorothy News Briefing"
    SubElement(channel, "link").text = SITE_URL
    SubElement(channel, "description").text = (
        "A 5-minute news briefing synthesized from 40+ sources across the political spectrum. "
        "Powered by Dorothy, the newspaper of averages."
    )
    SubElement(channel, "language").text = "en-us"
    SubElement(channel, "lastBuildDate").text = format_datetime(datetime.now(timezone.utc))
    SubElement(channel, "itunes:author").text = "Dorothy"
    SubElement(channel, "itunes:summary").text = (
        "Balanced news briefings from across the political spectrum."
    )

    category = SubElement(channel, "itunes:category")
    category.set("text", "News")
    SubElement(category, "itunes:category").set("text", "Daily News")

    SubElement(channel, "itunes:explicit").text = "no"

    for mp3_path in episodes:
        item = SubElement(channel, "item")

        # Parse date from filename: dorothy-YYYYMMDD-HHMM.mp3
        stem = mp3_path.stem  # dorothy-20260218-1400
        try:
            date_str = stem.replace("dorothy-", "")
            ep_date = datetime.strptime(date_str, "%Y%m%d-%H%M").replace(tzinfo=timezone.utc)
        except ValueError:
            ep_date = datetime.now(timezone.utc)

        title = f"Dorothy News Briefing — {ep_date.strftime('%B %d, %Y %I:%M %p UTC')}"
        SubElement(item, "title").text = title
        SubElement(item, "pubDate").text = format_datetime(ep_date)

        enclosure_url = f"{PODCAST_URL}/{mp3_path.name}"
        file_size = mp3_path.stat().st_size
        enclosure = SubElement(item, "enclosure")
        enclosure.set("url", enclosure_url)
        enclosure.set("length", str(file_size))
        enclosure.set("type", "audio/mpeg")

        SubElement(item, "guid").text = enclosure_url

        # Estimate duration from file size (128 kbps bitrate)
        duration_secs = int(file_size * 8 / 128000)
        minutes, seconds = divmod(duration_secs, 60)
        SubElement(item, "itunes:duration").text = f"{minutes}:{seconds:02d}"

        SubElement(item, "description").text = (
            f"Dorothy news briefing for {ep_date.strftime('%B %d, %Y')}. "
            "Top stories synthesized from sources across the political spectrum."
        )

    indent(rss, space="  ")
    tree = ElementTree(rss)
    tree.write(str(output_path), encoding="unicode", xml_declaration=True)

    logger.info("podcast_feed_generated", episodes=len(episodes), path=str(output_path))
    return output_path
