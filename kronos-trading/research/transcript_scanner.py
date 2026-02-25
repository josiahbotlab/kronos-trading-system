#!/usr/bin/env python3
"""
Kronos YouTube Transcript Scanner
===================================
Downloads auto-generated subtitles from @MoonDevOnYT using yt-dlp.
Supports one-shot batch scanning and periodic scheduled mode.

Priority ordering:
    Tier 1 (HIGH)  - MoltBot / OpenClaw videos
    Tier 2 (MED)   - General trading strategy content
    Tier 3 (LOW)   - Everything else

Usage:
    python -m research.transcript_scanner              # One-shot scan
    python -m research.transcript_scanner --scheduled  # Periodic (24h default)
    python -m research.transcript_scanner --max-videos 100
"""

import argparse
import json
import logging
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    print("Install deps: pip install yt-dlp")
    sys.exit(1)

from research.db import DB_PATH, init_research_db

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent.parent / "config" / "kronos.json"
TRANSCRIPT_DIR = Path(__file__).parent / "cache" / "transcripts" / "auto"

CHANNEL_URL = "https://www.youtube.com/@MoonDevOnYT/videos"

PRIORITY_KEYWORDS = {
    1: ["moltbot", "openclaw"],
    2: [
        "strategy", "trading", "backtest", "bot", "indicator",
        "signal", "entry", "exit", "profit", "liquidation",
        "cascade", "reversal", "momentum",
    ],
}

DEFAULT_SCAN_INTERVAL_HOURS = 24
DEFAULT_MAX_VIDEOS = 50
DELAY_BETWEEN_DOWNLOADS = 2  # seconds
COOKIES_PATH = Path(__file__).parent.parent / "config" / "youtube_cookies.txt"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("transcript_scanner")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load research config section from kronos.json."""
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("research", {}).get("transcript_scanner", {})
    except (json.JSONDecodeError, KeyError):
        return {}


def classify_priority(title: str) -> int:
    """Classify video priority tier based on title keywords.

    Returns:
        1 = MoltBot/OpenClaw content (highest priority)
        2 = General trading strategy content
        3 = Other/general content (lowest priority)
    """
    title_lower = title.lower()
    for tier in sorted(PRIORITY_KEYWORDS.keys()):
        for keyword in PRIORITY_KEYWORDS[tier]:
            if keyword in title_lower:
                return tier
    return 3


def _cookie_opts(cookies_path: Path | None = None) -> dict:
    """Return yt-dlp options for cookie authentication."""
    opts = {}
    path = cookies_path or COOKIES_PATH
    if path.exists():
        opts["cookiefile"] = str(path)
        log.info(f"Using cookies: {path}")
    return opts


def get_channel_video_list(channel_url: str, max_videos: int = DEFAULT_MAX_VIDEOS,
                           cookies_path: Path | None = None) -> list[dict]:
    """Fetch video metadata from channel without downloading.

    Args:
        channel_url: YouTube channel videos URL.
        max_videos: Maximum number of videos to retrieve.
        cookies_path: Optional path to Netscape-format cookies.txt.

    Returns:
        List of dicts with keys: video_id, title, upload_date, duration.
    """
    ydl_opts = {
        "extract_flat": True,
        "playlistend": max_videos,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        **_cookie_opts(cookies_path),
    }

    videos = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            if not info:
                log.error("Failed to extract channel info")
                return []

            entries = info.get("entries", [])
            for entry in entries:
                if entry is None:
                    continue
                videos.append({
                    "video_id": entry.get("id", ""),
                    "title": entry.get("title", "Unknown"),
                    "upload_date": entry.get("upload_date"),
                    "duration": entry.get("duration"),
                })
    except Exception as e:
        log.error(f"Error fetching channel video list: {e}")

    log.info(f"Found {len(videos)} videos on channel")
    return videos


def download_transcript(video_id: str, output_dir: Path,
                        cookies_path: Path | None = None) -> Path | None:
    """Download auto-generated subtitles for a video using yt-dlp.

    Writes plain text transcript to output_dir/{video_id}.txt

    Args:
        video_id: YouTube video ID.
        output_dir: Directory to write transcript file.
        cookies_path: Optional path to Netscape-format cookies.txt.

    Returns:
        Path to transcript .txt file, or None if no subtitles available.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "writeautomaticsub": True,
        "writesubtitles": True,
        "subtitlesformat": "vtt",
        "subtitleslangs": ["en"],
        "skip_download": True,
        "outtmpl": str(output_dir / "%(id)s"),
        "quiet": True,
        "no_warnings": True,
        **_cookie_opts(cookies_path),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        log.warning(f"yt-dlp error for {video_id}: {e}")
        return None

    # yt-dlp writes .en.vtt file — find it
    vtt_candidates = list(output_dir.glob(f"{video_id}*.vtt"))
    if not vtt_candidates:
        log.warning(f"No subtitles found for {video_id}")
        return None

    vtt_path = vtt_candidates[0]
    txt_path = output_dir / f"{video_id}.txt"

    try:
        clean_text = _vtt_to_text(vtt_path)
        if not clean_text.strip():
            log.warning(f"Empty transcript for {video_id}")
            vtt_path.unlink(missing_ok=True)
            return None

        txt_path.write_text(clean_text, encoding="utf-8")
        vtt_path.unlink(missing_ok=True)
        log.info(f"Transcript saved: {txt_path.name} ({len(clean_text)} chars)")
        return txt_path

    except Exception as e:
        log.error(f"Error processing VTT for {video_id}: {e}")
        vtt_path.unlink(missing_ok=True)
        return None


def _vtt_to_text(vtt_path: Path) -> str:
    """Convert VTT subtitle file to clean plaintext.

    Handles YouTube auto-caption deduplication (rolling captions repeat
    text from previous segments).
    """
    raw = vtt_path.read_text(encoding="utf-8")
    lines = raw.split("\n")

    # Skip VTT header
    text_lines = []
    seen = set()
    in_cue = False

    for line in lines:
        line = line.strip()

        # Skip empty lines, timestamps, and WEBVTT header
        if not line or line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            in_cue = False
            continue

        # Skip timestamp lines (e.g., "00:00:01.000 --> 00:00:04.000")
        if re.match(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->", line):
            in_cue = True
            continue

        # Skip cue position tags
        if re.match(r"^[\d]+$", line):
            continue

        if in_cue:
            # Strip HTML-like tags (e.g., <c>, </c>, <00:00:01.000>)
            clean = re.sub(r"<[^>]+>", "", line).strip()
            if clean and clean not in seen:
                seen.add(clean)
                text_lines.append(clean)

    return "\n".join(text_lines)


def _extract_keywords(transcript_text: str, title: str) -> list[str]:
    """Extract relevant trading keywords from transcript and title.

    Simple keyword extraction — not LLM-based.
    """
    trading_terms = {
        "liquidation", "cascade", "reversal", "momentum", "bollinger",
        "rsi", "ema", "sma", "macd", "stop loss", "take profit",
        "entry", "exit", "breakout", "support", "resistance",
        "scalping", "swing", "position size", "risk reward",
        "backtest", "drawdown", "sharpe", "bitcoin", "ethereum",
        "solana", "doge", "altcoin", "moltbot", "openclaw",
    }

    combined = f"{title} {transcript_text}".lower()
    found = [term for term in trading_terms if term in combined]
    return sorted(set(found))


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------
class TranscriptScanner:
    def __init__(
        self,
        db_path: Path = DB_PATH,
        transcript_dir: Path = TRANSCRIPT_DIR,
        config_path: Path = CONFIG_PATH,
    ):
        self.transcript_dir = transcript_dir
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.conn = init_research_db(db_path)
        self._running = True

        cfg = load_config(config_path)
        self.channel_url = cfg.get("channel_url", CHANNEL_URL)
        self.max_videos = cfg.get("max_videos_per_scan", DEFAULT_MAX_VIDEOS)
        self.scan_interval = cfg.get("scan_interval_hours", DEFAULT_SCAN_INTERVAL_HOURS)
        self.delay = cfg.get("delay_between_downloads", DELAY_BETWEEN_DOWNLOADS)
        self.subtitle_langs = cfg.get("subtitle_languages", ["en"])
        self.cookies_path = COOKIES_PATH

        # Merge config priority keywords with defaults
        tier1 = cfg.get("priority_keywords_tier1")
        tier2 = cfg.get("priority_keywords_tier2")
        if tier1:
            PRIORITY_KEYWORDS[1] = [k.lower() for k in tier1]
        if tier2:
            PRIORITY_KEYWORDS[2] = [k.lower() for k in tier2]

    def get_already_scanned_ids(self) -> set[str]:
        """Return set of video_ids already in the database."""
        rows = self.conn.execute("SELECT video_id FROM transcripts").fetchall()
        return {row["video_id"] for row in rows}

    def scan_channel(self) -> list[dict]:
        """Scan channel, download new transcripts, index in DB.

        Priority ordering: tier 1 videos processed first, then tier 2, then 3.

        Returns:
            List of dicts describing newly processed videos.
        """
        log.info(f"Scanning channel: {self.channel_url}")

        # 1. Get all videos from channel
        all_videos = get_channel_video_list(self.channel_url, self.max_videos,
                                            cookies_path=self.cookies_path)
        if not all_videos:
            log.warning("No videos found")
            return []

        # 2. Filter out already-scanned
        existing = self.get_already_scanned_ids()
        new_videos = [v for v in all_videos if v["video_id"] not in existing]
        log.info(f"{len(new_videos)} new videos to process (skipped {len(all_videos) - len(new_videos)} existing)")

        if not new_videos:
            return []

        # 3. Classify and sort by priority tier
        for v in new_videos:
            v["priority_tier"] = classify_priority(v["title"])

        new_videos.sort(key=lambda v: (v["priority_tier"], v.get("upload_date") or ""))

        tier_counts = {}
        for v in new_videos:
            tier_counts[v["priority_tier"]] = tier_counts.get(v["priority_tier"], 0) + 1
        log.info(f"Priority breakdown: {dict(sorted(tier_counts.items()))}")

        # 4. Process in priority order
        results = []
        for i, video in enumerate(new_videos):
            if not self._running:
                log.info("Shutdown requested, stopping scan")
                break

            success = self._process_video(video)
            if success:
                results.append(video)

            # Rate limit delay
            if i < len(new_videos) - 1:
                time.sleep(self.delay)

        log.info(f"Scan complete: {len(results)}/{len(new_videos)} transcripts downloaded")
        return results

    def _process_video(self, video_meta: dict) -> bool:
        """Download transcript and index a single video.

        Returns:
            True if successfully processed, False otherwise.
        """
        video_id = video_meta["video_id"]
        title = video_meta["title"]
        tier = video_meta["priority_tier"]
        tier_label = {1: "HIGH", 2: "MED", 3: "LOW"}.get(tier, "?")

        log.info(f"[{tier_label}] Processing: {title} ({video_id})")

        txt_path = download_transcript(video_id, self.transcript_dir,
                                       cookies_path=self.cookies_path)
        if txt_path is None:
            return False

        # Extract keywords
        transcript_text = txt_path.read_text(encoding="utf-8")
        keywords = _extract_keywords(transcript_text, title)

        # Index in database
        try:
            self.conn.execute(
                """INSERT OR IGNORE INTO transcripts
                   (video_id, title, channel, upload_date, source,
                    transcript_path, keywords, priority_tier, duration_seconds)
                   VALUES (?, ?, ?, ?, 'yt-dlp', ?, ?, ?, ?)""",
                (
                    video_id,
                    title,
                    "MoonDevOnYT",
                    video_meta.get("upload_date"),
                    str(txt_path),
                    json.dumps(keywords),
                    tier,
                    video_meta.get("duration"),
                ),
            )
            self.conn.commit()
            return True
        except Exception as e:
            log.error(f"DB error indexing {video_id}: {e}")
            return False

    def run_scheduled(self, interval_hours: float = DEFAULT_SCAN_INTERVAL_HOURS):
        """Run scanner on a periodic schedule."""
        log.info(f"Starting scheduled scanning (every {interval_hours}h)")
        interval_sec = interval_hours * 3600

        while self._running:
            try:
                results = self.scan_channel()
                log.info(f"Scheduled scan done: {len(results)} new transcripts")
            except Exception as e:
                log.error(f"Scan error: {e}")

            # Sleep in small increments so shutdown is responsive
            sleep_until = time.time() + interval_sec
            while self._running and time.time() < sleep_until:
                time.sleep(5)

    def shutdown(self):
        """Graceful shutdown."""
        self._running = False
        if self.conn:
            self.conn.close()
        log.info("Transcript scanner shut down")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Kronos YouTube Transcript Scanner")
    parser.add_argument("--scheduled", action="store_true",
                        help="Run on periodic schedule")
    parser.add_argument("--interval", type=float, default=DEFAULT_SCAN_INTERVAL_HOURS,
                        help=f"Hours between scans (default: {DEFAULT_SCAN_INTERVAL_HOURS})")
    parser.add_argument("--max-videos", type=int, default=DEFAULT_MAX_VIDEOS,
                        help=f"Max videos to scan per run (default: {DEFAULT_MAX_VIDEOS})")
    parser.add_argument("--cookies", default=None,
                        help="Path to Netscape-format cookies.txt for YouTube auth")
    parser.add_argument("--config", default=str(CONFIG_PATH),
                        help="Config file path")
    args = parser.parse_args()

    scanner = TranscriptScanner(config_path=Path(args.config))
    scanner.max_videos = args.max_videos
    if args.cookies:
        scanner.cookies_path = Path(args.cookies)

    def handle_signal(sig, frame):
        log.info(f"Signal {sig} received, shutting down...")
        scanner.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        if args.scheduled:
            scanner.run_scheduled(interval_hours=args.interval)
        else:
            results = scanner.scan_channel()
            print(f"\nScan complete: {len(results)} new transcript(s)")
            for r in results:
                tier_label = {1: "HIGH", 2: "MED", 3: "LOW"}.get(r["priority_tier"], "?")
                print(f"  [{tier_label}] {r['title']}")
    except KeyboardInterrupt:
        pass
    finally:
        scanner.shutdown()


if __name__ == "__main__":
    main()
