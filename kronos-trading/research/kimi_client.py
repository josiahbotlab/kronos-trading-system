#!/usr/bin/env python3
"""
Kronos Kimi LLM Client
========================
Calls Moonshot's Kimi API (OpenAI-compatible) for strategy extraction.

Uses the same endpoint and model as OpenClaw on the VPS:
    Endpoint: https://api.moonshot.ai/v1/chat/completions
    Model:    kimi-k2.5

API key is read from MOONSHOT_API_KEY environment variable.

Usage:
    export MOONSHOT_API_KEY=sk-...
    python -m research.strategy_extractor --process
"""

import json
import logging
import os
import urllib.request
import urllib.error

from research.strategy_extractor import BaseLLMClient, EXTRACTION_PROMPT

log = logging.getLogger("kimi_client")

API_URL = "https://api.moonshot.ai/v1/chat/completions"
MODEL = "kimi-k2.5"
MAX_TOKENS = 8192
TIMEOUT_SECONDS = 120


class KimiLLMClient(BaseLLMClient):
    """Kimi K2.5 via Moonshot API (OpenAI-compatible chat completions).

    Reads API key from MOONSHOT_API_KEY env var.
    """

    def __init__(self, api_key: str | None = None, model: str = MODEL):
        self.api_key = api_key or os.environ.get("MOONSHOT_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "MOONSHOT_API_KEY not set. "
                "Export it or pass api_key to KimiLLMClient()."
            )
        self.model = model
        log.info(f"KimiLLMClient initialized (model={self.model})")

    def extract_strategies(self, transcript_text: str, video_title: str) -> list[dict]:
        """Extract trading strategies via Kimi K2.5."""
        user_content = f"Title: {video_title}\n\nTranscript:\n{transcript_text}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            log.error(f"Kimi API HTTP {e.code}: {error_body}")
            raise RuntimeError(f"Kimi API error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            log.error(f"Kimi API connection error: {e.reason}")
            raise RuntimeError(f"Kimi API connection error: {e.reason}") from e

        # Parse response
        content = result["choices"][0]["message"]["content"]
        log.debug(f"Kimi raw response: {content[:200]}...")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(1))
            else:
                log.error(f"Could not parse Kimi response as JSON: {content[:200]}")
                return []

        strategies = parsed.get("strategies", [])

        # Log token usage
        usage = result.get("usage", {})
        if usage:
            log.info(
                f"Kimi tokens — prompt: {usage.get('prompt_tokens', '?')}, "
                f"completion: {usage.get('completion_tokens', '?')}, "
                f"total: {usage.get('total_tokens', '?')}"
            )

        return strategies
