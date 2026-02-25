#!/usr/bin/env python3
"""
Kronos Research Database
========================
Shared SQLite schema for transcript indexing and strategy extraction.

Tables:
    transcripts        - Video metadata + transcript file references
    extracted_strategies - LLM-extracted trading strategies from transcripts
"""

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger("research.db")

DB_PATH = Path(__file__).parent / "research.db"


def init_research_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize research database with schema.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        sqlite3.Connection with WAL mode enabled.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Performance pragmas (matches existing collector pattern)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA busy_timeout=5000")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            channel TEXT NOT NULL DEFAULT 'MoonDevOnYT',
            upload_date TEXT,
            source TEXT NOT NULL CHECK(source IN ('yt-dlp', 'manual')),
            processed_at TEXT NOT NULL DEFAULT (datetime('now')),
            transcript_path TEXT NOT NULL,
            keywords TEXT,
            priority_tier INTEGER NOT NULL DEFAULT 3,
            duration_seconds INTEGER,
            extraction_status TEXT NOT NULL DEFAULT 'pending'
                CHECK(extraction_status IN ('pending', 'processing', 'completed', 'failed')),
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS extracted_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_video_id TEXT NOT NULL,
            strategy_name TEXT NOT NULL,
            description TEXT,
            parameters TEXT,
            confidence REAL NOT NULL DEFAULT 0.0
                CHECK(confidence >= 0.0 AND confidence <= 1.0),
            category TEXT,
            raw_llm_response TEXT,
            extracted_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (source_video_id) REFERENCES transcripts(video_id)
        );

        CREATE INDEX IF NOT EXISTS idx_transcripts_video_id
            ON transcripts(video_id);
        CREATE INDEX IF NOT EXISTS idx_transcripts_source
            ON transcripts(source);
        CREATE INDEX IF NOT EXISTS idx_transcripts_priority
            ON transcripts(priority_tier);
        CREATE INDEX IF NOT EXISTS idx_transcripts_extraction_status
            ON transcripts(extraction_status);
        CREATE INDEX IF NOT EXISTS idx_strategies_source
            ON extracted_strategies(source_video_id);
        CREATE INDEX IF NOT EXISTS idx_strategies_confidence
            ON extracted_strategies(confidence);
    """)

    conn.commit()
    log.info(f"Research DB initialized: {db_path}")
    return conn
