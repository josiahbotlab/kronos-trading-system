"""
AI Swarm Validation Layer

Queries multiple AI models via OpenRouter for trade validation.
Each model provides independent analysis before trade execution.

Models:
- Claude 3.5 Sonnet: Technical analysis perspective
- GPT-4o-mini: Pattern recognition perspective
- DeepSeek Chat: Risk assessment perspective

Usage:
    from src.agents.ai_swarm import get_swarm_validation

    result = get_swarm_validation(
        symbol='AMD',
        gap_pct=5.2,
        direction='UP',
        crypto_conditions={'risk_level': 'NORMAL', 'liquidations_15m': 50000},
        regime='RANGING'
    )

    if result['should_trade']:
        # Execute with result['position_multiplier']
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from termcolor import cprint

sys.path.append('/Users/josiahgarcia/trading-bot')
load_dotenv()

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model configurations
SWARM_MODELS = {
    'technical': {
        'model': 'anthropic/claude-3.5-sonnet',
        'role': 'Technical Analysis Expert',
        'focus': 'chart patterns, support/resistance, momentum indicators'
    },
    'pattern': {
        'model': 'openai/gpt-4o-mini',
        'role': 'Pattern Recognition Specialist',
        'focus': 'gap patterns, historical behavior, statistical edge'
    },
    'risk': {
        'model': 'deepseek/deepseek-chat',
        'role': 'Risk Assessment Analyst',
        'focus': 'risk/reward ratio, position sizing, market conditions'
    }
}

# Timeout for API calls
API_TIMEOUT = 30  # seconds


def build_prompt(
    symbol: str,
    gap_pct: float,
    direction: str,
    crypto_conditions: Dict,
    regime: str,
    role: str,
    focus: str
) -> str:
    """Build the prompt for a swarm model."""

    crypto_risk = crypto_conditions.get('risk_level', 'UNKNOWN')
    liquidations = crypto_conditions.get('liquidations_15m_usd', 0)
    funding_sentiment = crypto_conditions.get('funding_sentiment', 'NEUTRAL')

    prompt = f"""You are a {role} for an automated Gap and Go trading system.
Your focus: {focus}

TRADE SETUP:
- Symbol: {symbol}
- Gap: {gap_pct:+.2f}% {direction}
- Market Regime: {regime}
- Crypto Market Risk: {crypto_risk}
- 15min Crypto Liquidations: ${liquidations:,.0f}
- Funding Sentiment: {funding_sentiment}
- Time: Pre-market gap trade (entry at market open)

GAP AND GO STRATEGY:
- Enter in direction of gap after market open
- Take profit: 5-8% from entry
- Stop loss: 7-15% from entry
- Session ends at 11:00 AM ET

ANALYZE THIS SETUP and provide your recommendation.

You MUST respond with ONLY a valid JSON object in this exact format:
{{"decision": "ENTER" or "SKIP" or "REDUCE", "confidence": 1-100, "reasoning": "Brief explanation (max 100 words)"}}

Decision meanings:
- ENTER: Take the trade with full position
- REDUCE: Take the trade with reduced position size
- SKIP: Do not take this trade

Respond with ONLY the JSON object, no other text."""

    return prompt


def query_model(
    model_key: str,
    model_config: Dict,
    symbol: str,
    gap_pct: float,
    direction: str,
    crypto_conditions: Dict,
    regime: str
) -> Dict:
    """Query a single model via OpenRouter."""

    if not OPENROUTER_API_KEY:
        return {
            'model': model_key,
            'model_id': model_config['model'],
            'decision': 'SKIP',
            'confidence': 0,
            'reasoning': 'OpenRouter API key not configured',
            'error': True
        }

    prompt = build_prompt(
        symbol=symbol,
        gap_pct=gap_pct,
        direction=direction,
        crypto_conditions=crypto_conditions,
        regime=regime,
        role=model_config['role'],
        focus=model_config['focus']
    )

    headers = {
        'Authorization': f'Bearer {OPENROUTER_API_KEY}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://github.com/trading-bot',
        'X-Title': 'Trading Bot AI Swarm'
    }

    payload = {
        'model': model_config['model'],
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.3,
        'max_tokens': 200
    }

    try:
        start_time = time.time()
        response = requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=API_TIMEOUT
        )
        elapsed = time.time() - start_time

        response.raise_for_status()
        data = response.json()

        # Extract the response content
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

        # Parse JSON from response
        try:
            # Try to extract JSON from the response
            # Handle cases where model might wrap JSON in markdown
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]

            result = json.loads(content.strip())

            return {
                'model': model_key,
                'model_id': model_config['model'],
                'decision': result.get('decision', 'SKIP').upper(),
                'confidence': result.get('confidence', 50),
                'reasoning': result.get('reasoning', 'No reasoning provided'),
                'elapsed_ms': int(elapsed * 1000),
                'error': False
            }

        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract decision from text
            content_upper = content.upper()
            if 'ENTER' in content_upper:
                decision = 'ENTER'
            elif 'REDUCE' in content_upper:
                decision = 'REDUCE'
            else:
                decision = 'SKIP'

            return {
                'model': model_key,
                'model_id': model_config['model'],
                'decision': decision,
                'confidence': 50,
                'reasoning': content[:200] if content else 'Failed to parse response',
                'elapsed_ms': int(elapsed * 1000),
                'error': False,
                'parse_warning': True
            }

    except requests.exceptions.Timeout:
        return {
            'model': model_key,
            'model_id': model_config['model'],
            'decision': 'SKIP',
            'confidence': 0,
            'reasoning': f'API timeout after {API_TIMEOUT}s',
            'error': True
        }
    except requests.exceptions.RequestException as e:
        return {
            'model': model_key,
            'model_id': model_config['model'],
            'decision': 'SKIP',
            'confidence': 0,
            'reasoning': f'API error: {str(e)[:100]}',
            'error': True
        }
    except Exception as e:
        return {
            'model': model_key,
            'model_id': model_config['model'],
            'decision': 'SKIP',
            'confidence': 0,
            'reasoning': f'Unexpected error: {str(e)[:100]}',
            'error': True
        }


def get_swarm_validation(
    symbol: str,
    gap_pct: float,
    direction: str,
    crypto_conditions: Dict,
    regime: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Query the AI swarm for trade validation.

    Args:
        symbol: Stock symbol (e.g., 'AMD')
        gap_pct: Gap percentage (e.g., 5.2 for +5.2%)
        direction: 'UP' or 'DOWN'
        crypto_conditions: Dict with risk_level, liquidations, etc.
        regime: Market regime ('BULL', 'BEAR', 'RANGING', 'HIGH_VOL')
        verbose: Whether to print status messages

    Returns:
        Dict with:
        - should_trade: bool
        - position_multiplier: float (1.0, 0.75, 0.5, or 0.0)
        - consensus: str ('STRONG_ENTER', 'ENTER', 'WEAK', 'SKIP')
        - responses: List of individual model responses
        - enter_count: Number of ENTER votes
        - timestamp: ISO timestamp
    """

    if verbose:
        cprint(f"\n[SWARM] Querying AI swarm for {symbol} trade validation...", "cyan")

    responses = []

    # Query all models in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                query_model,
                model_key,
                model_config,
                symbol,
                gap_pct,
                direction,
                crypto_conditions,
                regime
            ): model_key
            for model_key, model_config in SWARM_MODELS.items()
        }

        for future in as_completed(futures):
            model_key = futures[future]
            try:
                result = future.result()
                responses.append(result)
            except Exception as e:
                responses.append({
                    'model': model_key,
                    'model_id': SWARM_MODELS[model_key]['model'],
                    'decision': 'SKIP',
                    'confidence': 0,
                    'reasoning': f'Future error: {str(e)[:100]}',
                    'error': True
                })

    # Count votes
    enter_count = sum(1 for r in responses if r['decision'] == 'ENTER')
    reduce_count = sum(1 for r in responses if r['decision'] == 'REDUCE')
    skip_count = sum(1 for r in responses if r['decision'] == 'SKIP')

    # Determine consensus
    # 3/3 ENTER → full position
    # 2/3 ENTER → 75% position
    # 2/3 with mix of ENTER/REDUCE → 75% position
    # 1/3 ENTER → don't trade
    # 0/3 ENTER → don't trade

    if enter_count == 3:
        should_trade = True
        position_multiplier = 1.0
        consensus = 'STRONG_ENTER'
    elif enter_count == 2:
        should_trade = True
        position_multiplier = 0.75
        consensus = 'ENTER'
    elif enter_count == 1 and reduce_count >= 1:
        should_trade = True
        position_multiplier = 0.5
        consensus = 'WEAK'
    elif reduce_count >= 2:
        should_trade = True
        position_multiplier = 0.5
        consensus = 'REDUCE'
    else:
        should_trade = False
        position_multiplier = 0.0
        consensus = 'SKIP'

    # Log results
    if verbose:
        cprint(f"\n[SWARM] ═══════════════════════════════════════════", "magenta")
        cprint(f"[SWARM] AI SWARM VALIDATION RESULTS", "magenta", attrs=['bold'])
        cprint(f"[SWARM] ═══════════════════════════════════════════", "magenta")

        for r in responses:
            decision = r['decision']
            model_name = r['model'].upper()
            confidence = r.get('confidence', 0)
            reasoning = r.get('reasoning', '')[:80]
            elapsed = r.get('elapsed_ms', 0)

            if r.get('error'):
                color = 'red'
                icon = '✗'
            elif decision == 'ENTER':
                color = 'green'
                icon = '✓'
            elif decision == 'REDUCE':
                color = 'yellow'
                icon = '↓'
            else:
                color = 'red'
                icon = '✗'

            cprint(f"[SWARM] {icon} {model_name}: {decision} ({confidence}% confidence)", color)
            cprint(f"[SWARM]   └─ {reasoning}", "white")
            if elapsed:
                cprint(f"[SWARM]   └─ Response time: {elapsed}ms", "white")

        cprint(f"[SWARM] ───────────────────────────────────────────", "magenta")

        consensus_color = 'green' if should_trade else 'red'
        cprint(f"[SWARM] Consensus: {consensus} ({enter_count}/3 ENTER)", consensus_color, attrs=['bold'])
        cprint(f"[SWARM] Should Trade: {should_trade}", consensus_color)
        cprint(f"[SWARM] Position Multiplier: {position_multiplier:.0%}", consensus_color)
        cprint(f"[SWARM] ═══════════════════════════════════════════\n", "magenta")

    return {
        'should_trade': should_trade,
        'position_multiplier': position_multiplier,
        'consensus': consensus,
        'enter_count': enter_count,
        'reduce_count': reduce_count,
        'skip_count': skip_count,
        'responses': responses,
        'timestamp': datetime.now().isoformat()
    }


def test_swarm():
    """Test the swarm with a mock trade setup."""
    cprint("\n" + "=" * 60, "cyan")
    cprint("  AI SWARM VALIDATION TEST", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")

    # Mock trade setup
    symbol = 'AMD'
    gap_pct = 5.2
    direction = 'UP'
    crypto_conditions = {
        'risk_level': 'NORMAL',
        'liquidations_15m_usd': 45000,
        'funding_sentiment': 'NEUTRAL',
        'position_multiplier': 1.0
    }
    regime = 'RANGING'

    cprint(f"\n[TEST] Trade Setup:", "white")
    cprint(f"  Symbol: {symbol}", "white")
    cprint(f"  Gap: {gap_pct:+.2f}% {direction}", "white")
    cprint(f"  Regime: {regime}", "white")
    cprint(f"  Crypto Risk: {crypto_conditions['risk_level']}", "white")

    result = get_swarm_validation(
        symbol=symbol,
        gap_pct=gap_pct,
        direction=direction,
        crypto_conditions=crypto_conditions,
        regime=regime,
        verbose=True
    )

    cprint(f"\n[TEST] Final Result:", "cyan")
    cprint(f"  Should Trade: {result['should_trade']}", "white")
    cprint(f"  Position Multiplier: {result['position_multiplier']:.0%}", "white")
    cprint(f"  Consensus: {result['consensus']}", "white")

    return result


if __name__ == "__main__":
    test_swarm()
