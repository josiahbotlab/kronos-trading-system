#!/usr/bin/env python3
"""
Kronos Manual Transcript Ingestor
===================================
Picks up Glasp-exported transcript text files from the manual/ drop zone,
parses metadata, indexes in research.db, and moves to processed/.

Usage:
    python -m research.manual_ingest           # Ingest once then exit
    python -m research.manual_ingest --watch   # Watch directory continuously
"""

import argparse
import hashlib
import json
import logging
import re
import signal
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from research.db import DB_PATH, init_research_db

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MANUAL_DIR = Path(__file__).parent / "cache" / "transcripts" / "manual"
PROCESSED_DIR = Path(__file__).parent / "cache" / "transcripts" / "processed"
WATCH_INTERVAL_SECONDS = 30

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("manual_ingest")


# ---------------------------------------------------------------------------
# Glasp Parser
# ---------------------------------------------------------------------------
def parse_glasp_transcript(file_path: Path) -> dict:
    """Parse a Glasp-exported transcript file.

    Attempts to extract metadata from Glasp format headers.
    Falls back to filename-based metadata if headers are not present.

    Returns:
        Dict with keys: video_id, title, channel, upload_date, transcript_text
    """
    # Try multiple encodings
    text = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = file_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None or not text.strip():
        return {}

    result = {
        "video_id": None,
        "title": None,
        "channel": "unknown",
        "upload_date": None,
        "transcript_text": text,
    }

    # Try to extract YouTube video ID from content
    # Patterns: youtube.com/watch?v=ID, youtu.be/ID, /v/ID
    yt_patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/v/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in yt_patterns:
        match = re.search(pattern, text)
        if match:
            result["video_id"] = match.group(1)
            break

    # Fallback: generate deterministic ID from filename
    if not result["video_id"]:
        name_hash = hashlib.sha256(file_path.name.encode()).hexdigest()[:12]
        result["video_id"] = f"manual_{name_hash}"

    # Try to extract title from Glasp header lines
    lines = text.strip().split("\n")
    for i, line in enumerate(lines[:10]):  # Check first 10 lines
        line_stripped = line.strip()

        # Glasp often puts the title as the first non-empty line
        if i == 0 and line_stripped and not line_stripped.startswith("http"):
            result["title"] = line_stripped

        # Look for URL line to extract channel
        if "youtube.com/@" in line_stripped:
            channel_match = re.search(r"youtube\.com/@([^/\s]+)", line_stripped)
            if channel_match:
                result["channel"] = channel_match.group(1)

        # Look for date patterns
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", line_stripped)
        if date_match and not result["upload_date"]:
            result["upload_date"] = date_match.group(1)

    # Fallback title from filename
    if not result["title"]:
        result["title"] = file_path.stem.replace("_", " ").replace("-", " ").title()

    return result


def _extract_keywords(transcript_text: str, title: str) -> list[str]:
    """Extract relevant trading keywords from transcript and title."""
    trading_terms = {
        "liquidation", "cascade", "reversal", "momentum", "bollinger",
        "rsi", "ema", "sma", "macd", "stop loss", "take profit",
        "entry", "exit", "breakout", "support", "resistance",
        "scalping", "swing", "position size", "risk reward",
        "backtest", "drawdown", "sharpe", "bitcoin", "ethereum",
        "solana", "doge", "altcoin", "moltbot", "openclaw",
    }
    combined = f"{title} {transcript_text}".lower()
    return sorted(term for term in trading_terms if term in combined)


def _classify_priority(title: str) -> int:
    """Classify manual transcript priority from title."""
    title_lower = title.lower()
    for kw in ("moltbot", "openclaw"):
        if kw in title_lower:
            return 1
    trading_kw = [
        "strategy", "trading", "backtest", "bot", "indicator",
        "signal", "entry", "exit", "profit",
    ]
    for kw in trading_kw:
        if kw in title_lower:
            return 2
    return 3


# ---------------------------------------------------------------------------
# Ingestor
# ---------------------------------------------------------------------------
class ManualIngestor:
    def __init__(
        self,
        db_path: Path = DB_PATH,
        manual_dir: Path = MANUAL_DIR,
        processed_dir: Path = PROCESSED_DIR,
    ):
        self.manual_dir = manual_dir
        self.processed_dir = processed_dir
        self.manual_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.conn = init_research_db(db_path)
        self._running = True

    def scan_for_new_files(self) -> list[Path]:
        """Find unprocessed .txt files in the manual directory."""
        return sorted(self.manual_dir.glob("*.txt"))

    def ingest_file(self, file_path: Path) -> bool:
        """Process a single manual transcript file.

        Parses metadata, indexes in DB, moves to processed/.

        Returns:
            True if successfully ingested, False otherwise.
        """
        log.info(f"Ingesting: {file_path.name}")

        parsed = parse_glasp_transcript(file_path)
        if not parsed:
            log.warning(f"Empty or unreadable file: {file_path.name}")
            return False

        video_id = parsed["video_id"]

        # Check for duplicates
        existing = self.conn.execute(
            "SELECT id FROM transcripts WHERE video_id = ?", (video_id,)
        ).fetchone()
        if existing:
            log.info(f"Already indexed: {video_id} — moving to processed")
            self._move_to_processed(file_path)
            return False

        # Compute metadata
        title = parsed["title"]
        keywords = _extract_keywords(parsed["transcript_text"], title)
        priority = _classify_priority(title)
        tier_label = {1: "HIGH", 2: "MED", 3: "LOW"}.get(priority, "?")

        # Index in database
        try:
            self.conn.execute(
                """INSERT INTO transcripts
                   (video_id, title, channel, upload_date, source,
                    transcript_path, keywords, priority_tier)
                   VALUES (?, ?, ?, ?, 'manual', ?, ?, ?)""",
                (
                    video_id,
                    title,
                    parsed["channel"],
                    parsed.get("upload_date"),
                    str(file_path),
                    json.dumps(keywords),
                    priority,
                ),
            )
            self.conn.commit()
        except Exception as e:
            log.error(f"DB error for {video_id}: {e}")
            return False

        # Move to processed
        self._move_to_processed(file_path)

        log.info(f"[{tier_label}] Ingested: {title} ({video_id})")
        return True

    def _move_to_processed(self, file_path: Path):
        """Move file to processed/ with timestamp prefix."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = self.processed_dir / f"{ts}_{file_path.name}"
        shutil.move(str(file_path), str(dest))

    def ingest_all(self) -> int:
        """Scan and ingest all new manual transcript files.

        Returns:
            Number of files successfully ingested.
        """
        files = self.scan_for_new_files()
        if not files:
            log.info("No new manual transcripts found")
            return 0

        log.info(f"Found {len(files)} file(s) to ingest")
        count = 0
        for f in files:
            if self.ingest_file(f):
                count += 1
        return count

    def run_watch(self, interval_seconds: int = WATCH_INTERVAL_SECONDS):
        """Watch the manual directory for new files on a polling loop."""
        log.info(f"Watching {self.manual_dir} (every {interval_seconds}s)")

        while self._running:
            try:
                count = self.ingest_all()
                if count:
                    log.info(f"Watch cycle: ingested {count} file(s)")
            except Exception as e:
                log.error(f"Watch cycle error: {e}")

            # Sleep in small increments for responsive shutdown
            sleep_until = time.time() + interval_seconds
            while self._running and time.time() < sleep_until:
                time.sleep(1)

    def shutdown(self):
        """Graceful shutdown."""
        self._running = False
        if self.conn:
            self.conn.close()
        log.info("Manual ingestor shut down")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Kronos Manual Transcript Ingestor")
    parser.add_argument("--watch", action="store_true",
                        help="Watch directory continuously")
    parser.add_argument("--interval", type=int, default=WATCH_INTERVAL_SECONDS,
                        help=f"Seconds between watch scans (default: {WATCH_INTERVAL_SECONDS})")
    args = parser.parse_args()

    ingestor = ManualIngestor()

    def handle_signal(sig, frame):
        log.info(f"Signal {sig} received, shutting down...")
        ingestor.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        if args.watch:
            ingestor.run_watch(interval_seconds=args.interval)
        else:
            count = ingestor.ingest_all()
            print(f"\nIngested {count} manual transcript(s)")
    except KeyboardInterrupt:
        pass
    finally:
        ingestor.shutdown()


if __name__ == "__main__":
    main()
