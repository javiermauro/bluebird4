"""
Risk Regime Overlay - Crash Protection State Machine

Provides centralized risk management with three modes:
- NORMAL: Full trading with standard grid behavior
- RISK_OFF: Crash protection - blocks buys, cancels buy limits, blocks rebalance-down
- RECOVERY: Gradual re-entry with position ramping after stability confirmed

Triggers use 2-of-3 signals (momentum shock, ADX+direction, correlation spike).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import json
import os

from src.utils.atomic_io import atomic_write_json

# Project root for persistent state files (survives reboot, unlike /tmp)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_DIR = os.path.join(PROJECT_ROOT, "data", "state")

logger = logging.getLogger("RiskOverlay")


class RiskMode(Enum):
    """Risk regime states"""
    NORMAL = "NORMAL"
    RISK_OFF = "RISK_OFF"
    RECOVERY = "RECOVERY"


@dataclass
class RecoveryState:
    """Tracks recovery progress through stages"""
    stage: int = 0  # 0-3 corresponding to position ramp
    bars_in_stage: int = 0
    total_bars_in_recovery: int = 0
    prices_in_stage: List[float] = field(default_factory=list)


@dataclass
class BlockedAction:
    """Record of a blocked trading action"""
    timestamp: datetime
    symbol: str
    action_type: str  # "buy", "rebalance_down", "cancel_limit"
    notional: float
    reason: str


class RiskOverlay:
    """
    Risk Regime Overlay State Machine

    Controls trading behavior based on market conditions:
    - Blocks buys and rebalance-down during crashes (RISK_OFF)
    - Allows gradual re-entry after stability (RECOVERY)
    - Tracks telemetry in dollars to prove P&L impact
    """

    # State persistence file
    STATE_FILE = os.path.join(STATE_DIR, "risk-overlay.json")

    def __init__(self, config):
        self.config = config

        # Core state
        self.mode = RiskMode.NORMAL
        self.mode_entered_at = datetime.now()
        self.trigger_reasons: List[str] = []

        # Recovery tracking
        self.recovery_state = RecoveryState()

        # Telemetry - counts AND dollars
        self.telemetry = {
            "avoided_buys_count": 0,
            "avoided_buys_notional": 0.0,
            "cancelled_limits_count": 0,
            "cancelled_limits_notional": 0.0,
            "untracked_buys_count": 0,
            "rebalances_blocked_count": 0,
            "mode_transitions": [],
        }

        # Signal history for stability checks
        self.signal_history: List[Dict] = []
        self.price_history: Dict[str, List[float]] = {}  # symbol -> prices

        # Manual override
        self.manual_override: Optional[RiskMode] = None
        self.manual_override_at: Optional[datetime] = None

        # Load persisted state
        self._load_state()

        logger.info(f"RiskOverlay initialized in {self.mode.value} mode")

    def evaluate(self, signals: Dict) -> RiskMode:
        """
        Evaluate signals and potentially transition modes.

        Args:
            signals: Dict with keys:
                - momentum: float (e.g., -0.02 for -2%)
                - adx: float (e.g., 40)
                - adx_direction: str ("down", "up", "neutral")
                - max_correlation: float (e.g., 0.92)
                - current_prices: Dict[str, float] (symbol -> price)

        Returns:
            Current RiskMode after evaluation
        """
        if not self._is_enabled():
            return RiskMode.NORMAL

        # Manual override takes precedence
        if self.manual_override is not None:
            # IMPORTANT: If we're forcing a mode, we must also update self.mode,
            # otherwise the rest of the system (telemetry, API, transition hooks)
            # will believe we're still in the previous mode.
            if self.mode != self.manual_override:
                prev = self.mode
                self.mode = self.manual_override
                self.mode_entered_at = datetime.now()
                # Clear triggers when forcing NORMAL/RECOVERY; keep a clear reason string
                self.trigger_reasons = ["manual_override" + (f": {getattr(self, 'manual_override_reason', '')}" if getattr(self, 'manual_override_reason', None) else "")]
                # Reset recovery bookkeeping when leaving RECOVERY, and initialize when entering it
                if self.mode == RiskMode.RECOVERY:
                    self.recovery_state = RecoveryState(stage=0, bars_in_stage=0, total_bars_in_recovery=0)
                else:
                    self.recovery_state = RecoveryState()
                self._record_transition(prev, self.mode)
                self._save_state()
            return self.mode

        # Record signal history
        signals["timestamp"] = datetime.now()
        self.signal_history.append(signals)
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]

        # Update price history
        if "current_prices" in signals:
            for symbol, price in signals["current_prices"].items():
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                self.price_history[symbol].append(price)
                if len(self.price_history[symbol]) > 50:
                    self.price_history[symbol] = self.price_history[symbol][-50:]

        # State machine transitions
        previous_mode = self.mode

        if self.mode == RiskMode.NORMAL:
            self._evaluate_normal_to_risk_off(signals)
        elif self.mode == RiskMode.RISK_OFF:
            self._evaluate_risk_off_to_recovery(signals)
        elif self.mode == RiskMode.RECOVERY:
            self._evaluate_recovery(signals)

        # Log transitions
        if self.mode != previous_mode:
            self._record_transition(previous_mode, self.mode)

        return self.mode

    def _evaluate_normal_to_risk_off(self, signals: Dict) -> None:
        """Check if we should enter RISK_OFF from NORMAL"""
        triggers_fired = self._count_triggers(signals)

        required = getattr(self.config, "RISK_OFF_TRIGGERS_REQUIRED", 2)

        if triggers_fired >= required:
            self._enter_risk_off(signals)

    def _evaluate_risk_off_to_recovery(self, signals: Dict) -> None:
        """Check if we should enter RECOVERY from RISK_OFF"""
        # Must hold RISK_OFF for minimum time
        min_hold = getattr(self.config, "RISK_OFF_MIN_HOLD_MINUTES", 20)
        time_in_mode = (datetime.now() - self.mode_entered_at).total_seconds() / 60

        if time_in_mode < min_hold:
            return

        # Check for relapse (triggers firing again)
        triggers_fired = self._count_triggers(signals)
        if triggers_fired >= getattr(self.config, "RISK_OFF_TRIGGERS_REQUIRED", 2):
            # Reset hold timer
            self.mode_entered_at = datetime.now()
            self.trigger_reasons = self._get_trigger_reasons(signals)
            return

        # Check stability gate
        if self._check_stability_gate(signals):
            self._enter_recovery()

    def _evaluate_recovery(self, signals: Dict) -> None:
        """Manage RECOVERY stages and check for relapse or NORMAL transition"""
        # Check for relapse first
        triggers_fired = self._count_triggers(signals)
        if triggers_fired >= getattr(self.config, "RISK_OFF_TRIGGERS_REQUIRED", 2):
            self._enter_risk_off(signals, is_relapse=True)
            return

        # Update recovery state
        self.recovery_state.bars_in_stage += 1
        self.recovery_state.total_bars_in_recovery += 1

        # Track prices for "no new low" check
        if "current_prices" in signals:
            for price in signals["current_prices"].values():
                self.recovery_state.prices_in_stage.append(price)

        # Check if we can advance to next stage
        stability_bars = getattr(self.config, "RECOVERY_STABILITY_BARS", 10)

        if self.recovery_state.bars_in_stage >= stability_bars:
            if self._check_stage_advancement(signals):
                self._advance_recovery_stage()

    def _count_triggers(self, signals: Dict) -> int:
        """Count how many RISK_OFF triggers are firing"""
        count = 0

        # 1. Momentum shock
        momentum = signals.get("momentum", 0)
        momentum_threshold = getattr(self.config, "RISK_OFF_MOMENTUM_THRESHOLD", -0.015)
        if momentum < momentum_threshold:
            count += 1

        # 2. ADX + direction down (neutral doesn't count)
        adx = signals.get("adx", 0)
        adx_direction = signals.get("adx_direction", "neutral")
        adx_threshold = getattr(self.config, "RISK_OFF_ADX_THRESHOLD", 35)
        if adx > adx_threshold and adx_direction == "down":
            count += 1

        # 3. Correlation spike
        max_corr = signals.get("max_correlation", 0)
        corr_threshold = getattr(self.config, "RISK_OFF_CORRELATION_THRESHOLD", 0.90)
        if max_corr > corr_threshold:
            count += 1

        # 4. Drawdown velocity (optional, disabled by default)
        if getattr(self.config, "RISK_OFF_DRAWDOWN_VELOCITY_ENABLED", False):
            dd_velocity = signals.get("drawdown_velocity", 0)
            dd_threshold = getattr(self.config, "RISK_OFF_DRAWDOWN_VELOCITY", 0.02)
            if dd_velocity > dd_threshold:
                count += 1

        return count

    def _get_trigger_reasons(self, signals: Dict) -> List[str]:
        """Get human-readable trigger reasons"""
        reasons = []

        momentum = signals.get("momentum", 0)
        if momentum < getattr(self.config, "RISK_OFF_MOMENTUM_THRESHOLD", -0.015):
            reasons.append(f"momentum_shock={momentum:.2%}")

        adx = signals.get("adx", 0)
        adx_direction = signals.get("adx_direction", "neutral")
        if adx > getattr(self.config, "RISK_OFF_ADX_THRESHOLD", 35) and adx_direction == "down":
            reasons.append(f"adx_downtrend={adx:.1f}")

        max_corr = signals.get("max_correlation", 0)
        if max_corr > getattr(self.config, "RISK_OFF_CORRELATION_THRESHOLD", 0.90):
            reasons.append(f"correlation_spike={max_corr:.2f}")

        if getattr(self.config, "RISK_OFF_DRAWDOWN_VELOCITY_ENABLED", False):
            dd_velocity = signals.get("drawdown_velocity", 0)
            if dd_velocity > getattr(self.config, "RISK_OFF_DRAWDOWN_VELOCITY", 0.02):
                reasons.append(f"drawdown_velocity={dd_velocity:.2%}/hr")

        return reasons

    def _check_stability_gate(self, signals: Dict) -> bool:
        """Check if conditions are stable enough to enter RECOVERY"""
        stability_bars = getattr(self.config, "RECOVERY_STABILITY_BARS", 10)

        if len(self.signal_history) < stability_bars:
            return False

        recent_signals = self.signal_history[-stability_bars:]

        # Check momentum stability
        momentum_min = getattr(self.config, "RECOVERY_ENTRY_MOMENTUM_MIN", -0.005)
        for sig in recent_signals:
            if sig.get("momentum", -1) < momentum_min:
                return False

        # Check correlation easing
        corr_threshold = getattr(self.config, "RISK_OFF_CORRELATION_THRESHOLD", 0.90) - 0.05
        for sig in recent_signals:
            if sig.get("max_correlation", 1) > corr_threshold:
                return False

        # Check "no new low"
        if getattr(self.config, "RECOVERY_NO_NEW_LOW", True):
            if not self._check_no_new_low():
                return False

        return True

    def _check_no_new_low(self) -> bool:
        """Check that price hasn't made new lows during stability window"""
        stability_bars = getattr(self.config, "RECOVERY_STABILITY_BARS", 10)

        for symbol, prices in self.price_history.items():
            if len(prices) < stability_bars:
                continue

            recent = prices[-stability_bars:]
            current = recent[-1]
            min_price = min(recent)

            # Current price should be >= min of recent prices
            if current < min_price * 0.999:  # Small tolerance
                return False

        return True

    def _check_stage_advancement(self, signals: Dict) -> bool:
        """Check if we can advance to next RECOVERY stage"""
        # Need positive momentum to advance
        advance_momentum_min = getattr(self.config, "RECOVERY_ADVANCE_MOMENTUM_MIN", 0.0)

        # Check recent momentum
        stability_bars = getattr(self.config, "RECOVERY_STABILITY_BARS", 10)
        if len(self.signal_history) < stability_bars:
            return False

        recent_signals = self.signal_history[-stability_bars:]
        for sig in recent_signals:
            if sig.get("momentum", -1) < advance_momentum_min:
                return False

        # Check "no new low" for this stage
        if self.recovery_state.prices_in_stage:
            current = self.recovery_state.prices_in_stage[-1]
            min_price = min(self.recovery_state.prices_in_stage)
            if current < min_price * 0.999:
                return False

        return True

    def _enter_risk_off(self, signals: Dict, is_relapse: bool = False) -> None:
        """Transition to RISK_OFF mode"""
        self.mode = RiskMode.RISK_OFF
        self.mode_entered_at = datetime.now()
        self.trigger_reasons = self._get_trigger_reasons(signals)
        self.recovery_state = RecoveryState()  # Reset recovery state

        action = "RELAPSE" if is_relapse else "ENTRY"
        logger.warning(f"[RISK_OFF] {action}: {', '.join(self.trigger_reasons)}")

        self._save_state()

    def _enter_recovery(self) -> None:
        """Transition to RECOVERY mode"""
        self.mode = RiskMode.RECOVERY
        self.mode_entered_at = datetime.now()
        self.trigger_reasons = []
        self.recovery_state = RecoveryState(stage=0, bars_in_stage=0, total_bars_in_recovery=0)

        logger.info("[RECOVERY] Entering recovery mode at stage 0 (0.25x)")

        self._save_state()

    def _advance_recovery_stage(self) -> None:
        """Advance to next RECOVERY stage or NORMAL"""
        ramp = getattr(self.config, "RECOVERY_POSITION_RAMP", [0.25, 0.5, 0.75, 1.0])
        min_total_bars = getattr(self.config, "RECOVERY_MIN_TOTAL_BARS", 30)

        self.recovery_state.stage += 1
        self.recovery_state.bars_in_stage = 0
        self.recovery_state.prices_in_stage = []

        if self.recovery_state.stage >= len(ramp):
            # Check minimum total time in RECOVERY
            if self.recovery_state.total_bars_in_recovery >= min_total_bars:
                self._enter_normal()
            else:
                # Stay at max stage until min time met
                self.recovery_state.stage = len(ramp) - 1
                logger.info(f"[RECOVERY] At max stage, waiting for min bars ({self.recovery_state.total_bars_in_recovery}/{min_total_bars})")
        else:
            multiplier = ramp[self.recovery_state.stage]
            logger.info(f"[RECOVERY] Advanced to stage {self.recovery_state.stage} ({multiplier}x)")

        self._save_state()

    def _enter_normal(self) -> None:
        """Transition to NORMAL mode"""
        self.mode = RiskMode.NORMAL
        self.mode_entered_at = datetime.now()
        self.trigger_reasons = []
        self.recovery_state = RecoveryState()

        logger.info("[NORMAL] Full trading resumed")

        self._save_state()

    def _record_transition(self, from_mode: RiskMode, to_mode: RiskMode) -> None:
        """Record mode transition for telemetry"""
        self.telemetry["mode_transitions"].append({
            "timestamp": datetime.now().isoformat(),
            "from": from_mode.value,
            "to": to_mode.value,
            "reasons": self.trigger_reasons.copy(),
        })

        # Keep last 50 transitions
        if len(self.telemetry["mode_transitions"]) > 50:
            self.telemetry["mode_transitions"] = self.telemetry["mode_transitions"][-50:]

    # =========================================================================
    # Trading Gates
    # =========================================================================

    def allows_buy(self, symbol: str, notional: float = 0.0) -> Tuple[bool, str]:
        """
        Check if a buy order is allowed.

        Args:
            symbol: Trading symbol
            notional: Dollar value of the intended buy

        Returns:
            (allowed: bool, reason: str)
        """
        if not self._is_enabled():
            return True, "overlay_disabled"

        if self.mode == RiskMode.RISK_OFF:
            self._record_blocked_buy(symbol, notional, "risk_off_mode")
            return False, f"RISK_OFF: {', '.join(self.trigger_reasons)}"

        if self.mode == RiskMode.RECOVERY:
            # Buys allowed but with reduced size (handled by get_position_multiplier)
            return True, f"recovery_stage_{self.recovery_state.stage}"

        return True, "normal_mode"

    def allows_rebalance_down(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if downward rebalancing is allowed.

        CRITICAL: Blocked until NORMAL (not allowed in RECOVERY).

        Returns:
            (allowed: bool, reason: str)
        """
        if not self._is_enabled():
            return True, "overlay_disabled"

        if self.mode in (RiskMode.RISK_OFF, RiskMode.RECOVERY):
            self.telemetry["rebalances_blocked_count"] += 1
            return False, f"rebalance_down_blocked_in_{self.mode.value}"

        return True, "normal_mode"

    def allows_sell(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if a sell order is allowed.

        NOTE: Sells are ALWAYS allowed in RISK_OFF (override other filters).

        Returns:
            (allowed: bool, reason: str)
        """
        # Sells always allowed - this is for exit opportunities
        return True, "sells_always_allowed"

    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on current mode.

        Returns:
            0.0 in RISK_OFF (no buys)
            0.25-1.0 in RECOVERY (staged ramp)
            1.0 in NORMAL
        """
        if not self._is_enabled():
            return 1.0

        if self.mode == RiskMode.RISK_OFF:
            return 0.0

        if self.mode == RiskMode.RECOVERY:
            ramp = getattr(self.config, "RECOVERY_POSITION_RAMP", [0.25, 0.5, 0.75, 1.0])
            stage = min(self.recovery_state.stage, len(ramp) - 1)
            return ramp[stage]

        return 1.0

    # =========================================================================
    # Telemetry
    # =========================================================================

    def _record_blocked_buy(self, symbol: str, notional: float, reason: str) -> None:
        """Record a blocked buy for telemetry"""
        self.telemetry["avoided_buys_count"] += 1
        self.telemetry["avoided_buys_notional"] += notional

    def record_cancelled_limit(self, symbol: str, notional: float) -> None:
        """Record a cancelled limit order for telemetry"""
        self.telemetry["cancelled_limits_count"] += 1
        self.telemetry["cancelled_limits_notional"] += notional

    def record_untracked_buy(self, symbol: str, order_id: str) -> None:
        """Record an untracked buy that was NOT cancelled"""
        self.telemetry["untracked_buys_count"] += 1
        logger.warning(f"[RISK] Untracked BUY limit {order_id} for {symbol} - NOT cancelled")

    def get_status(self) -> Dict:
        """Get comprehensive status for API"""
        ramp = getattr(self.config, "RECOVERY_POSITION_RAMP", [0.25, 0.5, 0.75, 1.0])

        # Get latest signals for visual gauges
        latest_signals = self.signal_history[-1] if self.signal_history else {}

        return {
            "enabled": self._is_enabled(),
            "mode": self.mode.value,
            "mode_since": self.mode_entered_at.isoformat(),
            "mode_duration_minutes": (datetime.now() - self.mode_entered_at).total_seconds() / 60,
            "trigger_reasons": self.trigger_reasons,
            "position_multiplier": self.get_position_multiplier(),
            "recovery": {
                "stage": self.recovery_state.stage,
                "stage_multiplier": ramp[min(self.recovery_state.stage, len(ramp) - 1)] if self.mode == RiskMode.RECOVERY else None,
                "bars_in_stage": self.recovery_state.bars_in_stage,
                "total_bars_in_recovery": self.recovery_state.total_bars_in_recovery,
            } if self.mode == RiskMode.RECOVERY else None,
            "telemetry": {
                "avoided_buys_count": self.telemetry["avoided_buys_count"],
                "avoided_buys_notional": round(self.telemetry["avoided_buys_notional"], 2),
                "cancelled_limits_count": self.telemetry["cancelled_limits_count"],
                "cancelled_limits_notional": round(self.telemetry["cancelled_limits_notional"], 2),
                "untracked_buys_count": self.telemetry["untracked_buys_count"],
                "rebalances_blocked_count": self.telemetry["rebalances_blocked_count"],
            },
            "manual_override": self.manual_override.value if self.manual_override else None,
            "caps": {
                "normal_exposure": getattr(self.config, "NORMAL_TOTAL_EXPOSURE_CAP", 0.70),
                "risk_off_exposure": getattr(self.config, "RISK_OFF_TOTAL_EXPOSURE_CAP", 0.40),
            },
            # Raw signal values for visual threshold gauges
            "current_signals": {
                "momentum": latest_signals.get("momentum", 0),
                "correlation": latest_signals.get("max_correlation", 0),
                "adx": latest_signals.get("adx", 0),
                "adx_direction": latest_signals.get("adx_direction", "neutral"),
            },
            "thresholds": {
                "momentum": getattr(self.config, "RISK_OFF_MOMENTUM_THRESHOLD", -0.015),
                "correlation": getattr(self.config, "RISK_OFF_CORRELATION_THRESHOLD", 0.90),
                "adx": getattr(self.config, "RISK_OFF_ADX_THRESHOLD", 35),
            },
        }

    # =========================================================================
    # Manual Override
    # =========================================================================

    # Command file for API-initiated overrides
    COMMAND_FILE = os.path.join(STATE_DIR, "risk-overlay-command.json")

    def set_manual_override(self, mode: Optional[RiskMode], reason: str = None) -> None:
        """
        Set manual override mode.

        Args:
            mode: RiskMode to force, or None to clear override
            reason: Optional reason for the override
        """
        self.manual_override = mode
        self.manual_override_at = datetime.now() if mode else None
        self.manual_override_reason = reason

        if mode:
            logger.warning(f"[RISK] Manual override set to {mode.value}" + (f" - {reason}" if reason else ""))
        else:
            logger.info("[RISK] Manual override cleared")

        self._save_state()

    def check_command_file(self) -> bool:
        """
        Check for and process command file from API.

        Returns:
            True if a command was processed
        """
        try:
            if not os.path.exists(self.COMMAND_FILE):
                return False

            with open(self.COMMAND_FILE, "r") as f:
                command = json.load(f)

            # Delete the command file to prevent re-processing
            os.remove(self.COMMAND_FILE)

            action = command.get("action")
            reason = command.get("reason", "API override")

            if action == "set_override":
                mode_str = command.get("mode", "NORMAL")
                try:
                    mode = RiskMode(mode_str)
                    self.set_manual_override(mode, reason)
                    logger.warning(f"[RISK] API override applied: {mode.value}")
                    return True
                except ValueError:
                    logger.error(f"[RISK] Invalid mode in command file: {mode_str}")

            elif action == "clear_override":
                self.set_manual_override(None)
                logger.info("[RISK] API override cleared")
                return True

            return False

        except json.JSONDecodeError as e:
            logger.error(f"[RISK] Invalid JSON in command file: {e}")
            # Remove invalid command file
            try:
                os.remove(self.COMMAND_FILE)
            except:
                pass
            return False
        except Exception as e:
            logger.error(f"[RISK] Error processing command file: {e}")
            return False

    # =========================================================================
    # Persistence
    # =========================================================================

    def _is_enabled(self) -> bool:
        """Check if risk overlay is enabled"""
        return getattr(self.config, "RISK_OVERLAY_ENABLED", True)

    def _save_state(self) -> None:
        """Save state to disk for crash recovery"""
        try:
            state = {
                "mode": self.mode.value,
                "mode_entered_at": self.mode_entered_at.isoformat(),
                "trigger_reasons": self.trigger_reasons,
                "recovery_state": {
                    "stage": self.recovery_state.stage,
                    "bars_in_stage": self.recovery_state.bars_in_stage,
                    "total_bars_in_recovery": self.recovery_state.total_bars_in_recovery,
                },
                "telemetry": self.telemetry,
                "manual_override": self.manual_override.value if self.manual_override else None,
                "saved_at": datetime.now().isoformat(),
            }

            # Use atomic write to prevent corruption on power loss
            atomic_write_json(self.STATE_FILE, state)

        except Exception as e:
            logger.error(f"Failed to save risk overlay state: {e}")

    def _load_state(self) -> None:
        """Load state from disk"""
        try:
            if not os.path.exists(self.STATE_FILE):
                return

            with open(self.STATE_FILE, "r") as f:
                state = json.load(f)

            # Restore mode
            self.mode = RiskMode(state.get("mode", "NORMAL"))
            self.mode_entered_at = datetime.fromisoformat(state.get("mode_entered_at", datetime.now().isoformat()))
            self.trigger_reasons = state.get("trigger_reasons", [])

            # Restore recovery state
            rs = state.get("recovery_state", {})
            self.recovery_state = RecoveryState(
                stage=rs.get("stage", 0),
                bars_in_stage=rs.get("bars_in_stage", 0),
                total_bars_in_recovery=rs.get("total_bars_in_recovery", 0),
            )

            # Restore telemetry
            self.telemetry.update(state.get("telemetry", {}))

            # Restore manual override
            override = state.get("manual_override")
            self.manual_override = RiskMode(override) if override else None

            logger.info(f"Restored risk overlay state: {self.mode.value}")

        except Exception as e:
            logger.error(f"Failed to load risk overlay state: {e}")
