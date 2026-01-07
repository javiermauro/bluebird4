"""
SmartGridAdvisor - Grid adjustment recommendation engine.

Phase 1: Advisory-only (shadow mode)
- Monitors drift, fill rate, volatility
- Recommends adjustments but doesn't execute
- Respects existing rebalance mechanisms (range-break, overshoot)

This advisor does NOT create a third competing rebalance system.
It monitors and recommends, but existing mechanisms handle execution.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.strategy.grid_trading import GridOrderSide
from src.database import db as database

logger = logging.getLogger(__name__)


# Default thresholds (can be overridden by config)
DRIFT_TRIGGER_THRESHOLD = 0.55  # Recommend recenter when drift >= 55%
DRIFT_CLEAR_THRESHOLD = 0.40    # Clear recommendation when drift <= 40%
COOLDOWN_MINUTES = 60           # Min time between recommendations
EVAL_INTERVAL_SECONDS = 300     # 5 minutes (time-based throttle)
ATR_HISTORY_SIZE = 100          # Ring buffer size for ATR percentile
FILL_RATE_CACHE_TTL = 60        # Seconds to cache fill rate


class SmartGridAdvisor:
    """
    Grid adjustment recommendation engine.

    Phase 1: Advisory-only (shadow mode)
    - Monitors drift, fill rate, volatility
    - Recommends adjustments but doesn't execute
    - Respects existing rebalance mechanisms
    """

    STATE_FILE = Path("data/state/smart-grid-advisor.json")

    def __init__(self, config, grid_strategy, risk_overlay, orchestrator):
        """
        Initialize SmartGridAdvisor.

        Args:
            config: Bot configuration module
            grid_strategy: GridTradingStrategy instance
            risk_overlay: RiskOverlay instance
            orchestrator: Orchestrator instance
        """
        self.config = config
        self.grid_strategy = grid_strategy
        self.risk_overlay = risk_overlay
        self.orchestrator = orchestrator

        # Load config values with defaults
        self.drift_trigger = getattr(config, 'SMART_GRID_DRIFT_TRIGGER', DRIFT_TRIGGER_THRESHOLD)
        self.drift_clear = getattr(config, 'SMART_GRID_DRIFT_CLEAR', DRIFT_CLEAR_THRESHOLD)
        self.cooldown_minutes = getattr(config, 'SMART_GRID_COOLDOWN_MINUTES', COOLDOWN_MINUTES)
        self.eval_interval_seconds = getattr(config, 'SMART_GRID_EVAL_INTERVAL_SECONDS', EVAL_INTERVAL_SECONDS)
        self.min_spacing_pct = getattr(config, 'SMART_GRID_MIN_SPACING_PCT', 0.85)

        # State
        self.state = self._load_state()
        self._last_eval_at: Dict[str, datetime] = {}
        self._fill_rate_cache: Dict[str, tuple] = {}  # symbol -> (value, timestamp)

        logger.info(f"[SMARTGRID] Advisor initialized (shadow mode)")
        logger.info(f"[SMARTGRID] Drift trigger={self.drift_trigger:.0%}, clear={self.drift_clear:.0%}, cooldown={self.cooldown_minutes}min")

    def _load_state(self) -> dict:
        """Load state from file or create default."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE, 'r') as f:
                    state = json.load(f)
                    logger.info(f"[SMARTGRID] Loaded state from {self.STATE_FILE}")
                    return state
            except Exception as e:
                logger.warning(f"[SMARTGRID] Failed to load state: {e}, using defaults")

        return {
            'symbols': {},
            'atr_history': {},
            'saved_at': None
        }

    def _save_state(self) -> None:
        """Save state to file."""
        try:
            self.state['saved_at'] = datetime.now().isoformat()
            self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write
            tmp_file = self.STATE_FILE.with_suffix('.tmp')
            with open(tmp_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            tmp_file.replace(self.STATE_FILE)
        except Exception as e:
            logger.error(f"[SMARTGRID] Failed to save state: {e}")

    def _should_evaluate(self, symbol: str) -> bool:
        """Time-based throttle - max once per symbol per eval_interval_seconds."""
        now = datetime.now()
        last_eval = self._last_eval_at.get(symbol)

        if last_eval is None or (now - last_eval).total_seconds() >= self.eval_interval_seconds:
            self._last_eval_at[symbol] = now
            return True
        return False

    def _get_symbol_state(self, symbol: str) -> dict:
        """Get or create symbol state."""
        if 'symbols' not in self.state:
            self.state['symbols'] = {}

        if symbol not in self.state['symbols']:
            self.state['symbols'][symbol] = {
                'recommendation_active': False,
                'last_recommendation_at': None,
                'last_clear_at': None
            }

        return self.state['symbols'][symbol]

    def _compute_drift(self, current_price: float, grid_config) -> dict:
        """
        Compute price drift from grid center.

        Returns dict with:
            drift_pct: 0.0 to 1.0+ (0.5 = midpoint, 1.0 = at edge)
            direction: 'up' or 'down'
            current_price, grid_center, upper_price, lower_price
        """
        upper = grid_config.upper_price
        lower = grid_config.lower_price
        center = (upper + lower) / 2
        half_range = (upper - lower) / 2

        if half_range <= 0:
            return {
                'drift_pct': 0.0,
                'direction': 'neutral',
                'current_price': current_price,
                'grid_center': center,
                'upper_price': upper,
                'lower_price': lower
            }

        drift = (current_price - center) / half_range  # -1.0 to +1.0

        return {
            'drift_pct': abs(drift),
            'direction': 'up' if drift > 0 else 'down',
            'current_price': current_price,
            'grid_center': center,
            'upper_price': upper,
            'lower_price': lower
        }

    def _compute_level_imbalance(self, levels: List) -> dict:
        """
        Compute unfilled level imbalance.

        Returns dict with buy_pct, sell_pct, imbalanced flag.
        """
        unfilled_buys = [l for l in levels if not l.is_filled and l.side == GridOrderSide.BUY]
        unfilled_sells = [l for l in levels if not l.is_filled and l.side == GridOrderSide.SELL]
        total = len(unfilled_buys) + len(unfilled_sells)

        if total == 0:
            return {
                'buy_pct': 0.5,
                'sell_pct': 0.5,
                'unfilled_buys': 0,
                'unfilled_sells': 0,
                'imbalanced': False
            }

        buy_pct = len(unfilled_buys) / total
        sell_pct = len(unfilled_sells) / total

        return {
            'buy_pct': buy_pct,
            'sell_pct': sell_pct,
            'unfilled_buys': len(unfilled_buys),
            'unfilled_sells': len(unfilled_sells),
            'imbalanced': max(buy_pct, sell_pct) > 0.6
        }

    def _get_fill_rate(self, symbol: str, window_hours: float = 4.0) -> float:
        """
        Get fill rate (fills per hour) from database.

        Uses caching to avoid hot-path DB queries.
        """
        now = datetime.now()

        # Check cache
        if symbol in self._fill_rate_cache:
            cached_value, cached_at = self._fill_rate_cache[symbol]
            if (now - cached_at).total_seconds() < FILL_RATE_CACHE_TTL:
                return cached_value

        # Query database
        try:
            with database.get_db_connection() as conn:
                cursor = conn.cursor()
                cutoff = (now - timedelta(hours=window_hours)).isoformat()
                cursor.execute("""
                    SELECT COUNT(*) FROM trades
                    WHERE source='grid' AND symbol=? AND timestamp >= ?
                """, [symbol, cutoff])
                count = cursor.fetchone()[0]

            fill_rate = count / window_hours
            self._fill_rate_cache[symbol] = (fill_rate, now)
            return fill_rate

        except Exception as e:
            logger.warning(f"[SMARTGRID] Failed to get fill rate: {e}")
            return 0.0

    def _update_atr_history(self, symbol: str, atr_pct: float) -> float:
        """
        Update ATR history and compute percentile.

        Returns ATR percentile (0.0 to 1.0).
        """
        if 'atr_history' not in self.state:
            self.state['atr_history'] = {}

        if symbol not in self.state['atr_history']:
            self.state['atr_history'][symbol] = []

        history = self.state['atr_history'][symbol]

        # Add to ring buffer
        history.append(atr_pct)
        if len(history) > ATR_HISTORY_SIZE:
            history.pop(0)

        # Compute percentile
        if len(history) < 20:
            return 0.5  # Default until enough data

        return sum(1 for h in history if h <= atr_pct) / len(history)

    def _get_cached_atr_percentile(self, symbol: str) -> Optional[float]:
        """Get most recent ATR percentile from history."""
        history = self.state.get('atr_history', {}).get(symbol, [])
        if len(history) < 20:
            return None

        # Use last value in history
        last_atr = history[-1] if history else 0
        return sum(1 for h in history if h <= last_atr) / len(history)

    def _existing_rebalance_imminent(self, symbol: str, current_price: float,
                                      grid_config, bot) -> Optional[str]:
        """
        Check if existing rebalance mechanisms are about to fire.

        Returns reason string if imminent, None otherwise.
        Suppresses our recommendations when existing systems will handle it.
        """
        # Condition A: Outside-range (using config's rebalance_threshold)
        threshold = getattr(grid_config, 'rebalance_threshold', 0.02)

        # Check if we're 80% of the way to triggering range-break
        if current_price > grid_config.upper_price * (1 + threshold * 0.8):
            return "range_break_up_imminent"
        if current_price < grid_config.lower_price * (1 - threshold * 0.8):
            return "range_break_down_imminent"

        # Condition B: Overshoot counter active (only if mode is 'rebalance')
        overshoot_mode = getattr(bot.config, 'LIMIT_ORDER_OVERSHOOT_MODE', 'wait')
        if overshoot_mode == 'rebalance' and hasattr(bot, '_no_eligible_bars'):
            overshoot_bars = bot._no_eligible_bars.get(symbol, 0)
            bars_threshold = getattr(bot.config, 'OVERSHOOT_BARS_THRESHOLD', 5)
            if overshoot_bars >= bars_threshold - 1:
                return f"overshoot_pending_{overshoot_bars}_bars"

        return None

    def _check_gates(self, symbol: str, regime_metrics: dict, bot) -> dict:
        """
        Check all gating conditions.

        Returns dict with 'passed' bool and 'reason' if not passed.
        """
        # Gate 1: Risk Overlay must be NORMAL
        if self.risk_overlay:
            overlay_mode = self.risk_overlay.mode  # Access attribute directly
            if overlay_mode.value != 'NORMAL':
                return {
                    'passed': False,
                    'reason': f'risk_overlay_{overlay_mode.value}',
                    'gate': 'risk_overlay'
                }

        # Gate 2: Orchestrator must not be DEFENSIVE
        if self.orchestrator:
            orch_mode = self.orchestrator.last_mode.get(symbol)
            if orch_mode and orch_mode.value == 'defensive':
                return {
                    'passed': False,
                    'reason': 'orchestrator_defensive',
                    'gate': 'orchestrator'
                }

        # Gate 3: Regime should not be strong downtrend
        regime = regime_metrics.get('regime')
        adx = regime_metrics.get('adx', 0)
        if regime and hasattr(regime, 'value') and regime.value == 'TRENDING_DOWN' and adx >= 35:
            return {
                'passed': False,
                'reason': f'strong_downtrend_adx_{adx:.0f}',
                'gate': 'regime'
            }

        return {'passed': True, 'reason': None, 'gate': None}

    def _evaluate_drift_trigger(self, symbol: str, drift_pct: float) -> bool:
        """
        Evaluate drift trigger with hysteresis.

        Returns True if should recommend recenter.
        """
        symbol_state = self._get_symbol_state(symbol)
        now = datetime.now()

        if symbol_state['recommendation_active']:
            # Already recommending - check if should clear
            if drift_pct <= self.drift_clear:
                symbol_state['recommendation_active'] = False
                symbol_state['last_clear_at'] = now.isoformat()
                logger.info(f"[SMARTGRID] {symbol}: Cleared recommendation (drift={drift_pct:.1%})")
            return symbol_state['recommendation_active']
        else:
            # Not recommending - check if should trigger
            if drift_pct >= self.drift_trigger:
                # Check cooldown
                last_rec = symbol_state.get('last_recommendation_at')
                if last_rec:
                    last_rec_dt = datetime.fromisoformat(last_rec)
                    cooldown_elapsed = (now - last_rec_dt).total_seconds() / 60
                    if cooldown_elapsed < self.cooldown_minutes:
                        logger.debug(f"[SMARTGRID] {symbol}: Cooldown active ({cooldown_elapsed:.0f}/{self.cooldown_minutes}min)")
                        return False

                symbol_state['recommendation_active'] = True
                symbol_state['last_recommendation_at'] = now.isoformat()
                return True

            return False

    def evaluate(self, symbol: str, current_price: float,
                 regime_metrics: dict, bot) -> dict:
        """
        Evaluate if grid adjustment should be recommended.

        Called from handle_bar() but throttled to max once per eval_interval_seconds.
        Returns recommendation dict (always, even if no action).

        Args:
            symbol: Trading symbol (e.g., 'AVAX/USD')
            current_price: Current market price
            regime_metrics: Dict with regime, adx, atr_pct, correlation
            bot: Bot instance for accessing _no_eligible_bars etc.

        Returns:
            Dict with action, reason, drift, fill_rate, atr_percentile, etc.
        """
        # Guard: No grid loaded for this symbol
        if symbol not in self.grid_strategy.grids:
            return {
                'action': 'NONE',
                'reason': 'no_grid',
                'symbol': symbol,
                'would_execute': False
            }

        # Throttle: time-based
        if not self._should_evaluate(symbol):
            return {'action': 'SKIP', 'reason': 'throttled', 'symbol': symbol}

        # Check gates
        gate_result = self._check_gates(symbol, regime_metrics, bot)
        if not gate_result['passed']:
            return {
                'action': 'NONE',
                'reason': gate_result['reason'],
                'symbol': symbol,
                'would_execute': False,
                'gates': gate_result
            }

        # Get grid state
        grid_state = self.grid_strategy.grids[symbol]
        grid_config = grid_state.config

        # Compute metrics
        drift = self._compute_drift(current_price, grid_config)
        level_imbalance = self._compute_level_imbalance(grid_state.levels)
        fill_rate = self._get_fill_rate(symbol)
        atr_pct = regime_metrics.get('atr_pct', 0)
        atr_percentile = self._update_atr_history(symbol, atr_pct)

        # Check for existing rebalance imminent
        imminent = self._existing_rebalance_imminent(symbol, current_price, grid_config, bot)
        if imminent:
            return {
                'action': 'NONE',
                'reason': f'existing_rebalance_imminent:{imminent}',
                'symbol': symbol,
                'existing_rebalance_imminent': True,
                'imminent_type': imminent,
                'drift': drift,
                'level_imbalance': level_imbalance,
                'fill_rate': fill_rate,
                'atr_percentile': atr_percentile,
                'would_execute': False
            }

        # Evaluate drift trigger with hysteresis
        should_recenter = self._evaluate_drift_trigger(symbol, drift['drift_pct'])

        result = {
            'action': 'RECOMMEND_RECENTER' if should_recenter else 'NONE',
            'reason': f"drift={drift['drift_pct']:.1%}" if should_recenter else 'within_threshold',
            'symbol': symbol,
            'drift': drift,
            'level_imbalance': level_imbalance,
            'fill_rate': fill_rate,
            'atr_percentile': atr_percentile,
            'would_execute': False,  # Phase 1: always False
            'gates': gate_result,
            'recommendation_active': self._get_symbol_state(symbol)['recommendation_active']
        }

        # Log recommendation
        if should_recenter:
            logger.info(f"[SMARTGRID] {symbol} RECOMMEND_RECENTER: "
                       f"drift={drift['drift_pct']:.1%} {drift['direction']}, "
                       f"fill_rate={fill_rate:.2f}/hr")

        self._save_state()
        return result

    def evaluate_background(self, symbol: str, current_price: float) -> dict:
        """
        Lighter evaluation for background task (without bar data).

        Used by periodic task when WS bars stall.
        """
        # Guard: No grid loaded
        if symbol not in self.grid_strategy.grids:
            return {
                'action': 'NONE',
                'reason': 'no_grid',
                'symbol': symbol,
                'would_execute': False
            }

        # Skip throttle for background task (it has its own 300s timer)

        # Get grid state
        grid_state = self.grid_strategy.grids[symbol]
        grid_config = grid_state.config

        # Compute basic drift
        drift = self._compute_drift(current_price, grid_config)

        # Minimal gate check (just risk overlay)
        if self.risk_overlay:
            overlay_mode = self.risk_overlay.mode  # Access attribute directly
            if overlay_mode.value != 'NORMAL':
                return {
                    'action': 'NONE',
                    'reason': f'risk_overlay_{overlay_mode.value}',
                    'symbol': symbol,
                    'drift': drift,
                    'would_execute': False
                }

        # Check drift state (don't modify, just read)
        symbol_state = self._get_symbol_state(symbol)

        return {
            'action': 'RECOMMEND_RECENTER' if symbol_state['recommendation_active'] else 'NONE',
            'reason': f"drift={drift['drift_pct']:.1%}" if symbol_state['recommendation_active'] else 'within_threshold',
            'symbol': symbol,
            'drift': drift,
            'recommendation_active': symbol_state['recommendation_active'],
            'would_execute': False,
            'source': 'background_task'
        }

    def get_status(self) -> dict:
        """Get current status for API endpoint."""
        return {
            'enabled': getattr(self.config, 'SMART_GRID_ENABLED', False),
            'enforce': getattr(self.config, 'SMART_GRID_ENFORCE', False),
            'symbols': {
                symbol: {
                    'recommendation_active': self._get_symbol_state(symbol).get('recommendation_active', False),
                    'last_recommendation_at': self._get_symbol_state(symbol).get('last_recommendation_at'),
                    'last_clear_at': self._get_symbol_state(symbol).get('last_clear_at'),
                    'atr_percentile': self._get_cached_atr_percentile(symbol),
                }
                for symbol in self.grid_strategy.grids.keys()
            },
            'config': {
                'drift_trigger': self.drift_trigger,
                'drift_clear': self.drift_clear,
                'cooldown_minutes': self.cooldown_minutes,
                'eval_interval_seconds': self.eval_interval_seconds,
                'min_spacing_pct': self.min_spacing_pct
            },
            'state_file': str(self.STATE_FILE),
            'last_saved': self.state.get('saved_at')
        }
