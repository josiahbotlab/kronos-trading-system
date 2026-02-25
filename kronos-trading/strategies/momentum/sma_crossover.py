#!/usr/bin/env python3
"""
SMA Crossover Strategy (Test/Reference)
========================================
Simple moving average crossover for testing the backtester pipeline.
Not meant for production - just validates the infrastructure works.

Long when fast SMA > slow SMA, close when fast < slow.
"""

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal


class SMACrossover(BaseStrategy):
    name = "sma_crossover"
    version = "1.0"

    def default_params(self) -> dict:
        return {
            "fast_period": 10,
            "slow_period": 30,
            "max_history": 100,
        }

    def on_candle(self, candle: CandleData) -> Signal:
        fast_period = self.get_param("fast_period")
        slow_period = self.get_param("slow_period")

        fast = self.sma(fast_period)
        slow = self.sma(slow_period)

        if fast is None or slow is None:
            return Signal(direction=None)

        if fast > slow:
            return Signal(direction=1, tag="sma_bull")
        elif fast < slow:
            return Signal(direction=0, tag="sma_bear")

        return Signal(direction=None)
