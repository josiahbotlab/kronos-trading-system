"""
ML-Enhanced Market Regime Detection

Combines rule-based and machine learning approaches for regime classification:
- BULL: Forward 5-day return > 2%
- BEAR: Forward 5-day return < -2%
- RANGE: Forward 5-day return between -1% and 1%
- HIGH_VOL: Average daily move > 1.5%

Features used:
- 5, 10, 20, 50 day returns
- 5, 10, 20 day volatility (std of returns)
- RSI (14 period)
- Distance from 20/50/200 SMA (as percentage)
- Volume vs 20-day average
- ADX (trend strength)
- Number of up days in last 5, 10, 20 days

Usage:
    from src.agents.regime_agent import get_ml_regime, get_regime, RegimeAgent

    agent = RegimeAgent()
    ml_regime, confidence = agent.get_ml_regime('SPY')
    rule_regime = agent.get_rule_based_regime('SPY')

    # Compare both methods
    agent.compare_regimes('SPY')

    # Train/retrain the model
    agent.train_model()
"""

import os
import sys
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List, Literal
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings('ignore')

sys.path.append('/Users/josiahgarcia/trading-bot')
load_dotenv()

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: scikit-learn not available. ML features disabled.")

# Try to import ta library for indicators
try:
    from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator
    from ta.momentum import RSIIndicator
    from ta.volatility import AverageTrueRange
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: ta library not available. Using manual indicator calculations.")

# Alpaca imports
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Crypto market data imports
try:
    from src.data.liquidation_tracker import get_liquidation_totals
    LIQUIDATION_AVAILABLE = True
except ImportError:
    LIQUIDATION_AVAILABLE = False
    print("Warning: liquidation_tracker not available")

try:
    from src.data.market_indicators import get_market_snapshot
    MARKET_INDICATORS_AVAILABLE = True
except ImportError:
    MARKET_INDICATORS_AVAILABLE = False
    print("Warning: market_indicators not available")

# Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Paths
MODEL_PATH = Path('/Users/josiahgarcia/trading-bot/models/regime_classifier.pkl')
SCALER_PATH = Path('/Users/josiahgarcia/trading-bot/models/regime_scaler.pkl')

# Regime types
RegimeType = Literal['BULL', 'BEAR', 'RANGE', 'HIGH_VOL']

# Label thresholds for training
LABEL_THRESHOLDS = {
    'bull_threshold': 0.03,      # Forward 5-day return > 3%
    'bear_threshold': -0.03,     # Forward 5-day return < -3%
    'range_upper': 0.015,        # Between -1.5% and +1.5%
    'range_lower': -0.015,
    'high_vol_threshold': 0.025  # Average daily move > 2.5% (very high)
}


class RegimeAgent:
    """ML-Enhanced Regime Detection Agent."""

    def __init__(self):
        self.api = self._get_api()
        self.model = None
        self.scaler = None
        self.feature_names = []
        self._load_model()

    def _get_api(self):
        """Get Alpaca API instance."""
        if not ALPACA_AVAILABLE:
            return None
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            return None
        base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
        return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

    def _load_model(self):
        """Load trained model and scaler if available."""
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            try:
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"Loaded regime model from {MODEL_PATH}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
                self.scaler = None

    def _save_model(self):
        """Save trained model and scaler."""
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved regime model to {MODEL_PATH}")

    def fetch_data(self, symbol: str = 'SPY', days: int = 750) -> pd.DataFrame:
        """Fetch historical data for training/prediction."""
        if not self.api:
            raise RuntimeError("Alpaca API not available")

        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days + 50)

        bars = self.api.get_bars(
            symbol,
            '1Day',
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            feed='iex'
        ).df

        if bars.empty:
            raise ValueError(f"No data returned for {symbol}")

        bars = bars.reset_index()
        bars.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
        bars = bars[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        bars = bars.sort_values('timestamp').reset_index(drop=True)

        return bars

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features for regime classification."""
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # 1. Returns (5, 10, 20, 50 day)
        features['return_5d'] = close.pct_change(5)
        features['return_10d'] = close.pct_change(10)
        features['return_20d'] = close.pct_change(20)
        features['return_50d'] = close.pct_change(50)

        # 2. Volatility (5, 10, 20 day std of daily returns)
        daily_returns = close.pct_change()
        features['vol_5d'] = daily_returns.rolling(5).std()
        features['vol_10d'] = daily_returns.rolling(10).std()
        features['vol_20d'] = daily_returns.rolling(20).std()

        # 3. RSI (14 period)
        if TA_AVAILABLE:
            features['rsi'] = RSIIndicator(close, window=14).rsi()
        else:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))

        # 4. Distance from SMAs (as percentage)
        if TA_AVAILABLE:
            sma_20 = SMAIndicator(close, window=20).sma_indicator()
            sma_50 = SMAIndicator(close, window=50).sma_indicator()
            sma_200 = SMAIndicator(close, window=200).sma_indicator()
        else:
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean()

        features['dist_sma_20'] = (close - sma_20) / sma_20 * 100
        features['dist_sma_50'] = (close - sma_50) / sma_50 * 100
        features['dist_sma_200'] = (close - sma_200) / sma_200 * 100

        # 5. Volume vs 20-day average
        vol_avg_20 = volume.rolling(20).mean()
        features['volume_ratio'] = volume / vol_avg_20

        # 6. ADX (trend strength)
        if TA_AVAILABLE:
            features['adx'] = ADXIndicator(high, low, close, window=14).adx()
        else:
            # Manual ADX calculation
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()

            plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            features['adx'] = dx.rolling(14).mean()

        # 7. Number of up days in last 5, 10, 20 days
        up_days = (daily_returns > 0).astype(int)
        features['up_days_5'] = up_days.rolling(5).sum()
        features['up_days_10'] = up_days.rolling(10).sum()
        features['up_days_20'] = up_days.rolling(20).sum()

        # 8. Average absolute daily move (for HIGH_VOL detection)
        features['avg_daily_move'] = abs(daily_returns).rolling(5).mean()

        # Store feature names
        self.feature_names = features.columns.tolist()

        return features

    def create_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """Create training labels based on forward returns."""
        close = df['close']

        # Forward 5-day return
        forward_return = close.shift(-5) / close - 1

        # Calculate rolling volatility for HIGH_VOL detection
        # Use ATR-based approach instead of avg daily move
        daily_range = (df['high'] - df['low']) / close
        avg_daily_range = daily_range.rolling(20).mean()
        forward_range = daily_range.shift(-1).rolling(5).mean().shift(-4)

        labels = pd.Series(index=df.index, dtype=str)

        # Priority: Use return-based first, only HIGH_VOL for extreme volatility
        for i in range(len(df)):
            fwd_ret = forward_return.iloc[i] if i < len(forward_return) else np.nan
            fwd_range = forward_range.iloc[i] if i < len(forward_range) else np.nan
            avg_range = avg_daily_range.iloc[i] if i < len(avg_daily_range) else np.nan

            if pd.isna(fwd_ret):
                labels.iloc[i] = np.nan
            elif fwd_ret > LABEL_THRESHOLDS['bull_threshold']:
                labels.iloc[i] = 'BULL'
            elif fwd_ret < LABEL_THRESHOLDS['bear_threshold']:
                labels.iloc[i] = 'BEAR'
            elif not pd.isna(fwd_range) and not pd.isna(avg_range) and fwd_range > avg_range * 2:
                # Only HIGH_VOL if forward range is 2x the average
                labels.iloc[i] = 'HIGH_VOL'
            else:
                labels.iloc[i] = 'RANGE'

        return labels

    def train_model(self, symbol: str = 'SPY', days: int = 750):
        """Train the regime classifier on historical data."""
        if not ML_AVAILABLE:
            print("scikit-learn not available. Cannot train model.")
            return

        print(f"Training regime classifier on {symbol} ({days} days)...")

        # Fetch data
        df = self.fetch_data(symbol, days)
        print(f"Fetched {len(df)} bars")

        # Calculate features
        features = self.calculate_features(df)

        # Create labels
        labels = self.create_labels(df, features)

        # Combine and drop NaN
        data = pd.concat([features, labels.rename('label')], axis=1)
        data = data.dropna()

        print(f"Training samples: {len(data)}")
        print(f"Label distribution:")
        print(data['label'].value_counts())

        # Split features and labels
        X = data[self.feature_names]
        y = data['label']

        # Train/test split (last 6 months = ~126 trading days for testing)
        test_size = min(126, int(len(X) * 0.2))
        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

        print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n{'='*50}")
        print(f"OUT-OF-SAMPLE RESULTS (last {test_size} days)")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 Feature Importance:")
        print(importance.head(10).to_string(index=False))

        # Save model
        self._save_model()

        return accuracy

    def get_ml_regime(self, symbol: str = 'SPY') -> Tuple[RegimeType, float]:
        """
        Get ML-based regime prediction with confidence score.

        Returns:
            Tuple of (regime, confidence)
        """
        if not self.model or not self.scaler:
            print("Model not trained. Training now...")
            self.train_model()

        # Fetch recent data
        df = self.fetch_data(symbol, days=250)

        # Calculate features
        features = self.calculate_features(df)

        # Get latest row
        latest_features = features.iloc[-1:][self.feature_names]

        if latest_features.isna().any().any():
            # Not enough data for all features
            return 'RANGE', 0.0

        # Scale and predict
        X_scaled = self.scaler.transform(latest_features)
        prediction = self.model.predict(X_scaled)[0]

        # Get confidence from probability
        proba = self.model.predict_proba(X_scaled)[0]
        max_proba = max(proba)
        confidence = max_proba * 100

        return prediction, confidence

    def get_rule_based_regime(self, symbol: str = 'SPY') -> Tuple[RegimeType, float]:
        """
        Get rule-based regime prediction (original method).

        Uses SMA50, SMA200, ADX, ATR.
        """
        df = self.fetch_data(symbol, days=250)

        close = df['close']
        high = df['high']
        low = df['low']

        # Calculate indicators
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()

        # ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()

        avg_atr = atr.iloc[-20:].mean()

        # Get latest values
        current_price = close.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        current_sma_200 = sma_200.iloc[-1]
        current_adx = adx.iloc[-1]
        current_atr = atr.iloc[-1]
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

        # Determine regime
        if atr_ratio > 2.0:
            return 'HIGH_VOL', min(100, int((atr_ratio - 1) * 50))
        elif current_price > current_sma_50 and current_sma_50 > current_sma_200 and current_adx > 25:
            return 'BULL', min(100, int(current_adx * 2))
        elif current_price < current_sma_50 and current_sma_50 < current_sma_200 and current_adx > 25:
            return 'BEAR', min(100, int(current_adx * 2))
        elif current_adx < 20:
            return 'RANGE', min(100, int((20 - current_adx) * 5))
        elif current_price > current_sma_50:
            return 'BULL', 30
        else:
            return 'BEAR', 30

    def get_regime(self, symbol: str = 'SPY') -> Dict[str, Any]:
        """
        Get combined regime analysis from both ML and rule-based methods.

        Returns dict with:
            - ml_regime: ML prediction
            - ml_confidence: ML confidence
            - rule_regime: Rule-based prediction
            - rule_confidence: Rule-based confidence
            - disagreement: Whether methods disagree
            - final_regime: Combined recommendation
        """
        ml_regime, ml_conf = self.get_ml_regime(symbol)
        rule_regime, rule_conf = self.get_rule_based_regime(symbol)

        disagreement = ml_regime != rule_regime

        # Final regime: prefer ML if confidence > 60%, otherwise use rule-based
        if ml_conf > 60:
            final_regime = ml_regime
        else:
            final_regime = rule_regime

        return {
            'symbol': symbol,
            'ml_regime': ml_regime,
            'ml_confidence': round(ml_conf, 1),
            'rule_regime': rule_regime,
            'rule_confidence': round(rule_conf, 1),
            'disagreement': disagreement,
            'final_regime': final_regime,
            'timestamp': datetime.now().isoformat()
        }

    def compare_regimes(self, symbol: str = 'SPY'):
        """Pretty print comparison of ML vs rule-based regimes."""
        result = self.get_regime(symbol)

        print(f"\n{'='*60}")
        print(f"  REGIME ANALYSIS: {symbol}")
        print(f"{'='*60}")

        print(f"\n  ML-Based Regime:       {result['ml_regime']:<10} ({result['ml_confidence']:.1f}%)")
        print(f"  Rule-Based Regime:     {result['rule_regime']:<10} ({result['rule_confidence']:.1f}%)")

        if result['disagreement']:
            print(f"\n  WARNING: Methods DISAGREE")
        else:
            print(f"\n  Methods AGREE")

        print(f"\n  Final Recommendation:  {result['final_regime']}")
        print(f"\n{'='*60}\n")

        return result

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from trained model."""
        if not self.model:
            return None

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


# Convenience functions
def get_ml_regime(symbol: str = 'SPY') -> Tuple[RegimeType, float]:
    """Get ML-based regime prediction."""
    agent = RegimeAgent()
    return agent.get_ml_regime(symbol)


def get_regime(symbol: str = 'SPY') -> Dict[str, Any]:
    """Get combined regime analysis."""
    agent = RegimeAgent()
    return agent.get_regime(symbol)


# ─────────────────────────────────────────────────────────────
# CRYPTO MARKET RISK INTEGRATION
# ─────────────────────────────────────────────────────────────

def get_crypto_risk_level() -> Dict[str, Any]:
    """
    Assess crypto market risk based on liquidations and funding.

    High liquidations indicate market stress and potential cascading effects.
    Extreme funding rates indicate overleveraged market conditions.

    Returns:
        dict with:
        - risk_level: 'NORMAL', 'ELEVATED_RISK', 'HIGH_RISK', or 'UNKNOWN'
        - position_multiplier: Suggested position size multiplier (0.5-1.0)
        - risk_factors: List of detected risk factors
        - liquidations_15m_usd: Total liquidations in last 15 minutes
        - funding_sentiment: Funding rate sentiment
    """
    try:
        risk_factors = []
        liq_total = 0
        funding_sentiment = 'UNKNOWN'

        # Get liquidation data
        if LIQUIDATION_AVAILABLE:
            try:
                liq_15m = get_liquidation_totals('15 minutes')
                liq_total = liq_15m.get('total_usd', 0)
                long_pct = liq_15m.get('long_pct', 50)
                short_pct = liq_15m.get('short_pct', 50)

                # Check liquidation levels
                if liq_total > 10_000_000:  # $10M+ in 15min
                    risk_factors.append(f'HIGH liquidations: ${liq_total:,.0f} in 15min')
                elif liq_total > 5_000_000:  # $5M+ in 15min
                    risk_factors.append(f'ELEVATED liquidations: ${liq_total:,.0f} in 15min')
                elif liq_total > 1_000_000:  # $1M+ in 15min
                    risk_factors.append(f'Active liquidations: ${liq_total:,.0f} in 15min')

                # Check liquidation imbalance
                if long_pct > 70:
                    risk_factors.append(f'Longs getting liquidated ({long_pct:.0f}%) - bearish pressure')
                elif short_pct > 70:
                    risk_factors.append(f'Shorts getting liquidated ({short_pct:.0f}%) - bullish pressure')

            except Exception as e:
                risk_factors.append(f'Liquidation data error: {e}')

        # Get funding sentiment
        if MARKET_INDICATORS_AVAILABLE:
            try:
                snapshot = get_market_snapshot()
                funding_sentiment = snapshot.get('funding_sentiment', 'UNKNOWN')
                avg_funding = snapshot.get('avg_funding_yearly', 0)
                btc_funding = snapshot.get('btc_funding_yearly', 0)
                eth_funding = snapshot.get('eth_funding_yearly', 0)

                # Check funding extremes
                if funding_sentiment == 'EXTREME_GREED':
                    risk_factors.append(f'EXTREME_GREED funding ({avg_funding:+.1f}% yearly) - overleveraged longs')
                elif funding_sentiment == 'EXTREME_FEAR':
                    risk_factors.append(f'EXTREME_FEAR funding ({avg_funding:+.1f}% yearly) - overleveraged shorts')
                elif funding_sentiment == 'GREED':
                    risk_factors.append(f'GREED funding ({avg_funding:+.1f}% yearly) - elevated long leverage')
                elif funding_sentiment == 'FEAR':
                    risk_factors.append(f'FEAR funding ({avg_funding:+.1f}% yearly) - elevated short leverage')

            except Exception as e:
                risk_factors.append(f'Funding data error: {e}')

        # Determine overall risk level and position multiplier
        if liq_total > 10_000_000 or funding_sentiment in ['EXTREME_GREED', 'EXTREME_FEAR']:
            risk_level = 'HIGH_RISK'
            position_multiplier = 0.5  # Reduce position size by 50%
        elif liq_total > 5_000_000 or funding_sentiment in ['GREED', 'FEAR']:
            risk_level = 'ELEVATED_RISK'
            position_multiplier = 0.75  # Reduce position size by 25%
        elif liq_total > 1_000_000:
            risk_level = 'CAUTIOUS'
            position_multiplier = 0.9  # Slight reduction
        else:
            risk_level = 'NORMAL'
            position_multiplier = 1.0  # Full position size

        return {
            'risk_level': risk_level,
            'position_multiplier': position_multiplier,
            'risk_factors': risk_factors,
            'liquidations_15m_usd': liq_total,
            'funding_sentiment': funding_sentiment,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Error getting crypto risk: {e}")
        return {
            'risk_level': 'UNKNOWN',
            'position_multiplier': 1.0,
            'risk_factors': [f'Error fetching crypto data: {e}'],
            'liquidations_15m_usd': 0,
            'funding_sentiment': 'UNKNOWN',
            'timestamp': datetime.now().isoformat()
        }


def get_full_regime_status(symbol: str = 'SPY') -> Dict[str, Any]:
    """
    Combined regime status including stock ML model and crypto risk.

    Provides a comprehensive view of market conditions by combining:
    1. Stock market regime (from ML model and rule-based detection)
    2. Crypto market risk (from liquidations and funding rates)

    Returns:
        dict with full market status including:
        - Stock regime and confidence
        - Crypto risk level and factors
        - Recommended position multiplier
        - BTC/ETH prices and funding rates
    """
    # Get existing stock regime (from ML model)
    try:
        agent = RegimeAgent()
        stock_regime = agent.get_regime(symbol)
    except Exception as e:
        print(f"Error getting stock regime: {e}")
        stock_regime = {
            'final_regime': 'UNKNOWN',
            'ml_confidence': 0,
            'rule_confidence': 0
        }

    # Get crypto risk
    crypto_risk = get_crypto_risk_level()

    # Get market snapshot for additional context
    btc_price = 0
    eth_price = 0
    btc_funding = 0
    eth_funding = 0

    if MARKET_INDICATORS_AVAILABLE:
        try:
            snapshot = get_market_snapshot()
            btc_price = snapshot.get('btc_price', 0)
            eth_price = snapshot.get('eth_price', 0)
            btc_funding = snapshot.get('btc_funding_yearly', 0)
            eth_funding = snapshot.get('eth_funding_yearly', 0)
        except Exception as e:
            print(f"Error getting market snapshot: {e}")

    return {
        # Stock regime
        'stock_symbol': symbol,
        'stock_regime': stock_regime.get('final_regime', 'UNKNOWN'),
        'stock_ml_regime': stock_regime.get('ml_regime', 'UNKNOWN'),
        'stock_rule_regime': stock_regime.get('rule_regime', 'UNKNOWN'),
        'stock_ml_confidence': stock_regime.get('ml_confidence', 0),
        'stock_rule_confidence': stock_regime.get('rule_confidence', 0),

        # Crypto risk
        'crypto_risk_level': crypto_risk['risk_level'],
        'crypto_liquidations_15m': crypto_risk['liquidations_15m_usd'],
        'crypto_funding_sentiment': crypto_risk['funding_sentiment'],
        'crypto_risk_factors': crypto_risk['risk_factors'],

        # Trading recommendation
        'recommended_position_multiplier': crypto_risk['position_multiplier'],

        # Crypto market data
        'btc_price': btc_price,
        'eth_price': eth_price,
        'btc_funding_yearly': btc_funding,
        'eth_funding_yearly': eth_funding,

        'timestamp': datetime.now().isoformat()
    }


def print_full_regime_status(symbol: str = 'SPY'):
    """Print a formatted full regime status."""
    status = get_full_regime_status(symbol)

    print(f"\n{'='*70}")
    print(f"  FULL MARKET REGIME STATUS")
    print(f"{'='*70}")

    # Stock Regime Section
    print(f"\n  STOCK MARKET ({status['stock_symbol']}):")
    print(f"  {'─'*60}")

    regime_colors = {
        'BULL': 'green', 'BEAR': 'red', 'RANGE': 'yellow',
        'HIGH_VOL': 'magenta', 'UNKNOWN': 'white'
    }

    print(f"    Final Regime:     {status['stock_regime']}")
    print(f"    ML Regime:        {status['stock_ml_regime']} ({status['stock_ml_confidence']:.1f}% confidence)")
    print(f"    Rule Regime:      {status['stock_rule_regime']} ({status['stock_rule_confidence']:.1f}% confidence)")

    # Crypto Risk Section
    print(f"\n  CRYPTO MARKET RISK:")
    print(f"  {'─'*60}")

    risk_level = status['crypto_risk_level']
    if risk_level == 'HIGH_RISK':
        print(f"    Risk Level:       {risk_level} ⚠️")
    elif risk_level == 'ELEVATED_RISK':
        print(f"    Risk Level:       {risk_level}")
    else:
        print(f"    Risk Level:       {risk_level}")

    print(f"    Liquidations:     ${status['crypto_liquidations_15m']:,.0f} (15min)")
    print(f"    Funding:          {status['crypto_funding_sentiment']}")

    if status['crypto_risk_factors']:
        print(f"\n    Risk Factors:")
        for factor in status['crypto_risk_factors']:
            print(f"      • {factor}")

    # Trading Recommendation
    print(f"\n  TRADING RECOMMENDATION:")
    print(f"  {'─'*60}")

    multiplier = status['recommended_position_multiplier']
    if multiplier < 0.75:
        print(f"    Position Size:    {multiplier:.0%} of normal (REDUCE RISK)")
    elif multiplier < 1.0:
        print(f"    Position Size:    {multiplier:.0%} of normal (CAUTIOUS)")
    else:
        print(f"    Position Size:    {multiplier:.0%} of normal")

    # Crypto Prices
    if status['btc_price'] > 0:
        print(f"\n  CRYPTO PRICES:")
        print(f"  {'─'*60}")
        print(f"    BTC:              ${status['btc_price']:,.2f}  (Funding: {status['btc_funding_yearly']:+.2f}%/yr)")
        print(f"    ETH:              ${status['eth_price']:,.2f}  (Funding: {status['eth_funding_yearly']:+.2f}%/yr)")

    print(f"\n{'='*70}\n")

    return status


def main():
    """Main entry point."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description='ML-Enhanced Regime Detection')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to analyze')
    parser.add_argument('--train', action='store_true', help='Train/retrain the model')
    parser.add_argument('--compare', action='store_true', help='Compare ML vs rule-based')
    parser.add_argument('--importance', action='store_true', help='Show feature importance')
    parser.add_argument('--full-status', action='store_true', help='Show full regime status with crypto risk')
    parser.add_argument('--crypto-risk', action='store_true', help='Show crypto risk level only')
    parser.add_argument('--live-test', action='store_true', help='Start trackers and show live full status')
    args = parser.parse_args()

    # Handle crypto-specific commands
    if args.crypto_risk:
        risk = get_crypto_risk_level()
        print(f"\n{'='*50}")
        print(f"  CRYPTO RISK LEVEL")
        print(f"{'='*50}")
        print(f"  Risk Level:         {risk['risk_level']}")
        print(f"  Position Multiplier: {risk['position_multiplier']:.0%}")
        print(f"  Liquidations (15m): ${risk['liquidations_15m_usd']:,.0f}")
        print(f"  Funding Sentiment:  {risk['funding_sentiment']}")
        if risk['risk_factors']:
            print(f"\n  Risk Factors:")
            for factor in risk['risk_factors']:
                print(f"    • {factor}")
        print(f"{'='*50}\n")
        return

    if args.full_status:
        print_full_regime_status(args.symbol)
        return

    if args.live_test:
        # Start trackers and show live full status
        print("Starting crypto trackers for live test...")

        try:
            from src.data.liquidation_tracker import start_tracker as start_liq_tracker, stop_tracker as stop_liq_tracker
            from src.data.market_indicators import start_market_indicators, stop_market_indicators

            # Start trackers
            start_liq_tracker(background=True)
            start_market_indicators(background=True)

            print("Waiting 10 seconds for data to populate...")
            time.sleep(10)

            # Show full status
            print("\n" + "="*70)
            print("  LIVE FULL REGIME STATUS")
            print("="*70)
            print_full_regime_status(args.symbol)

            # Cleanup
            stop_liq_tracker()
            stop_market_indicators()
            print("Trackers stopped.")

        except ImportError as e:
            print(f"Error: Could not import trackers: {e}")
        except Exception as e:
            print(f"Error during live test: {e}")
        return

    # Original functionality
    agent = RegimeAgent()

    if args.train:
        agent.train_model(args.symbol)

    if args.compare or (not args.train and not args.importance and not args.full_status):
        agent.compare_regimes(args.symbol)

    if args.importance:
        importance = agent.get_feature_importance()
        if importance is not None:
            print("\nFeature Importance:")
            print(importance.to_string(index=False))
        else:
            print("Model not trained. Run with --train first.")


if __name__ == '__main__':
    main()
