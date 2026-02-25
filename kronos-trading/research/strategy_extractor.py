#!/usr/bin/env python3
"""
Kronos Strategy Extractor
==========================
Feeds transcripts through an LLM to extract trading strategies.
The LLM client is pluggable — implement BaseLLMClient with your preferred provider.

Usage:
    python -m research.strategy_extractor --process      # Extract from pending transcripts
    python -m research.strategy_extractor --list          # List extracted strategies
    python -m research.strategy_extractor --summary       # Print extraction summary

Plugging in your LLM:
    from research.strategy_extractor import StrategyExtractor, BaseLLMClient

    class MyClient(BaseLLMClient):
        def extract_strategies(self, transcript_text, video_title):
            # Call your LLM here, return list of strategy dicts
            ...

    extractor = StrategyExtractor(llm_client=MyClient())
    extractor.process_all_pending()
"""

import argparse
import json
import logging
import signal
from abc import ABC, abstractmethod
from pathlib import Path

from research.db import DB_PATH, init_research_db

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent.parent / "config" / "kronos.json"
MAX_TRANSCRIPT_CHARS = 50000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("strategy_extractor")

# ---------------------------------------------------------------------------
# Extraction prompt (reusable by custom LLM clients)
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """You are a trading strategy extraction assistant. Analyze the following
YouTube video transcript from a crypto trading channel.

Extract any concrete trading strategies mentioned, including:
- Strategy name or description
- Entry/exit rules
- Indicators used (SMA, EMA, RSI, Bollinger Bands, liquidation data, etc.)
- Timeframes mentioned
- Risk management rules (stop loss, take profit, position sizing)
- Any specific parameters or thresholds

Return a JSON object with key "strategies" containing a list of objects, each with:
- strategy_name: descriptive name
- description: 1-3 sentence summary of the strategy logic
- parameters: object with any specific numbers/thresholds mentioned
- confidence: 0.0-1.0 how clearly the strategy was described
- category: one of "momentum", "reversal", "scalping", "mean_reversion", "breakout", "other"

If no concrete strategies are found, return {"strategies": []}.
"""


# ---------------------------------------------------------------------------
# Pluggable LLM Interface
# ---------------------------------------------------------------------------
class BaseLLMClient(ABC):
    """Abstract interface for LLM-based strategy extraction.

    Implement this class to wire up your preferred LLM (OpenAI, Anthropic,
    local model, etc.).

    Example with Anthropic:
        class AnthropicClient(BaseLLMClient):
            def __init__(self, api_key: str):
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)

            def extract_strategies(self, transcript_text, video_title):
                response = self.client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=4096,
                    system=EXTRACTION_PROMPT,
                    messages=[{"role": "user",
                               "content": f"Title: {video_title}\\n\\nTranscript:\\n{transcript_text}"}],
                )
                return json.loads(response.content[0].text)["strategies"]
    """

    @abstractmethod
    def extract_strategies(self, transcript_text: str, video_title: str) -> list[dict]:
        """Extract trading strategies from a transcript.

        Args:
            transcript_text: Full transcript text.
            video_title: Title of the source video (provides context).

        Returns:
            List of dicts, each with keys:
                - strategy_name: str
                - description: str
                - parameters: dict
                - confidence: float (0.0 to 1.0)
                - category: str
        """
        ...


class PlaceholderLLMClient(BaseLLMClient):
    """Placeholder LLM client — returns empty results.

    Replace with your preferred LLM integration by implementing BaseLLMClient.
    """

    def extract_strategies(self, transcript_text: str, video_title: str) -> list[dict]:
        log.warning(
            "PlaceholderLLMClient in use. No strategies extracted. "
            "Set MOONSHOT_API_KEY env var or pass llm_client to StrategyExtractor."
        )
        return []


def _default_llm_client() -> BaseLLMClient:
    """Auto-detect the best available LLM client.

    Uses KimiLLMClient if MOONSHOT_API_KEY is set, otherwise PlaceholderLLMClient.
    """
    import os
    if os.environ.get("MOONSHOT_API_KEY"):
        from research.kimi_client import KimiLLMClient
        return KimiLLMClient()
    log.warning("MOONSHOT_API_KEY not set — falling back to PlaceholderLLMClient")
    return PlaceholderLLMClient()


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
class StrategyExtractor:
    def __init__(
        self,
        db_path: Path = DB_PATH,
        llm_client: BaseLLMClient | None = None,
        config_path: Path = CONFIG_PATH,
    ):
        self.conn = init_research_db(db_path)
        self.llm = llm_client or _default_llm_client()

        # Load config
        self.max_chars = MAX_TRANSCRIPT_CHARS
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                ext_cfg = cfg.get("research", {}).get("strategy_extractor", {})
                self.max_chars = ext_cfg.get("max_transcript_chars", MAX_TRANSCRIPT_CHARS)
            except (json.JSONDecodeError, KeyError):
                pass

    def get_pending_transcripts(self) -> list[dict]:
        """Get transcripts that haven't been processed by the LLM yet.

        Returns rows where extraction_status = 'pending', ordered by
        priority_tier ASC (highest priority first).
        """
        rows = self.conn.execute(
            """SELECT video_id, title, transcript_path, priority_tier
               FROM transcripts
               WHERE extraction_status = 'pending'
               ORDER BY priority_tier ASC, processed_at ASC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def extract_from_transcript(self, transcript_row: dict) -> list[dict]:
        """Run LLM extraction on a single transcript.

        Returns:
            List of extracted strategy dicts.
        """
        video_id = transcript_row["video_id"]
        title = transcript_row["title"]
        transcript_path = Path(transcript_row["transcript_path"])

        if not transcript_path.exists():
            log.error(f"Transcript file not found: {transcript_path}")
            self._update_extraction_status(video_id, "failed")
            return []

        transcript_text = transcript_path.read_text(encoding="utf-8")

        # Truncate if needed
        if len(transcript_text) > self.max_chars:
            log.info(f"Truncating transcript {video_id} from {len(transcript_text)} to {self.max_chars} chars")
            transcript_text = transcript_text[:self.max_chars]

        self._update_extraction_status(video_id, "processing")

        try:
            strategies = self.llm.extract_strategies(transcript_text, title)
            raw_response = json.dumps(strategies, indent=2)
            self._save_strategies(video_id, strategies, raw_response)
            self._update_extraction_status(video_id, "completed")
            log.info(f"Extracted {len(strategies)} strategies from: {title}")
            return strategies

        except Exception as e:
            log.error(f"LLM extraction failed for {video_id}: {e}")
            self._update_extraction_status(video_id, "failed")
            return []

    def process_all_pending(self) -> int:
        """Process all pending transcripts through the LLM.

        Returns:
            Number of transcripts processed.
        """
        pending = self.get_pending_transcripts()
        if not pending:
            log.info("No pending transcripts to process")
            return 0

        log.info(f"Processing {len(pending)} pending transcript(s)")
        count = 0
        for row in pending:
            self.extract_from_transcript(row)
            count += 1

        return count

    def _save_strategies(self, video_id: str, strategies: list[dict], raw_response: str):
        """Save extracted strategies to the database."""
        for s in strategies:
            try:
                self.conn.execute(
                    """INSERT INTO extracted_strategies
                       (source_video_id, strategy_name, description,
                        parameters, confidence, category, raw_llm_response)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        video_id,
                        s.get("strategy_name", "Unknown"),
                        s.get("description", ""),
                        json.dumps(s.get("parameters", {})),
                        float(s.get("confidence", 0.0)),
                        s.get("category", "other"),
                        raw_response,
                    ),
                )
            except Exception as e:
                log.error(f"Error saving strategy from {video_id}: {e}")
        self.conn.commit()

    def _update_extraction_status(self, video_id: str, status: str):
        """Update extraction_status for a transcript."""
        self.conn.execute(
            "UPDATE transcripts SET extraction_status = ? WHERE video_id = ?",
            (status, video_id),
        )
        self.conn.commit()

    def get_extracted_strategies(
        self,
        min_confidence: float = 0.0,
        category: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query extracted strategies from the database."""
        query = """
            SELECT es.*, t.title as video_title, t.priority_tier
            FROM extracted_strategies es
            JOIN transcripts t ON es.source_video_id = t.video_id
            WHERE es.confidence >= ?
        """
        params: list = [min_confidence]

        if category:
            query += " AND es.category = ?"
            params.append(category)

        query += " ORDER BY es.confidence DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def print_summary(self):
        """Print a summary of extraction results."""
        # Transcript stats
        total = self.conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
        pending = self.conn.execute(
            "SELECT COUNT(*) FROM transcripts WHERE extraction_status = 'pending'"
        ).fetchone()[0]
        completed = self.conn.execute(
            "SELECT COUNT(*) FROM transcripts WHERE extraction_status = 'completed'"
        ).fetchone()[0]
        failed = self.conn.execute(
            "SELECT COUNT(*) FROM transcripts WHERE extraction_status = 'failed'"
        ).fetchone()[0]

        # Strategy stats
        strat_count = self.conn.execute("SELECT COUNT(*) FROM extracted_strategies").fetchone()[0]
        high_conf = self.conn.execute(
            "SELECT COUNT(*) FROM extracted_strategies WHERE confidence >= 0.7"
        ).fetchone()[0]

        # Category breakdown
        categories = self.conn.execute(
            """SELECT category, COUNT(*) as cnt
               FROM extracted_strategies
               GROUP BY category
               ORDER BY cnt DESC"""
        ).fetchall()

        print("\n=== Kronos Research Summary ===")
        print(f"\nTranscripts: {total} total")
        print(f"  Pending:   {pending}")
        print(f"  Completed: {completed}")
        print(f"  Failed:    {failed}")
        print(f"\nStrategies Extracted: {strat_count}")
        print(f"  High Confidence (>=70%): {high_conf}")

        if categories:
            print("\n  By Category:")
            for row in categories:
                print(f"    {row['category']}: {row['cnt']}")

        # Top strategies
        top = self.get_extracted_strategies(min_confidence=0.5, limit=5)
        if top:
            print("\n  Top Strategies:")
            for s in top:
                print(f"    [{s['confidence']:.0%}] {s['strategy_name']} ({s['category']})")
                if s.get("description"):
                    print(f"          {s['description'][:80]}")

        print()

    def shutdown(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
        log.info("Strategy extractor shut down")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Kronos Strategy Extractor")
    parser.add_argument("--process", action="store_true",
                        help="Process all pending transcripts")
    parser.add_argument("--list", action="store_true",
                        help="List extracted strategies")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Minimum confidence for --list (default: 0.5)")
    parser.add_argument("--summary", action="store_true",
                        help="Print extraction summary")
    args = parser.parse_args()

    extractor = StrategyExtractor()

    def handle_signal(sig, frame):
        extractor.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        if args.process:
            count = extractor.process_all_pending()
            print(f"\nProcessed {count} transcript(s)")
        elif args.list:
            strategies = extractor.get_extracted_strategies(
                min_confidence=args.min_confidence
            )
            if not strategies:
                print("No strategies found.")
            else:
                for s in strategies:
                    print(f"  [{s['confidence']:.0%}] {s['strategy_name']} ({s['category']})")
                    if s.get("description"):
                        print(f"        {s['description'][:100]}")
                    print(f"        Source: {s['video_title']}")
                    print()
        elif args.summary:
            extractor.print_summary()
        else:
            parser.print_help()
    finally:
        extractor.shutdown()


if __name__ == "__main__":
    main()
