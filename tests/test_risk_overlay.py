"""
Unit tests for Risk Regime Overlay.

Tests the crash protection state machine, including:
- Multi-signal triggering (2-of-3 signals)
- RISK_OFF buy blocking
- RISK_OFF sell allowing
- Rebalance-down blocking
- RECOVERY stage progression
- Relapse handling
- Position multiplier ramping
- Telemetry tracking
- State persistence
- Manual override
"""

import pytest
import sys
import os
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy.risk_overlay import RiskOverlay, RiskMode, RecoveryState


class MockConfig:
    """Mock configuration for testing."""

    def __init__(self, **kwargs):
        # Defaults
        self.RISK_OVERLAY_ENABLED = kwargs.get('RISK_OVERLAY_ENABLED', True)
        self.RISK_OFF_TRIGGERS_REQUIRED = kwargs.get('RISK_OFF_TRIGGERS_REQUIRED', 2)
        self.RISK_OFF_MIN_HOLD_MINUTES = kwargs.get('RISK_OFF_MIN_HOLD_MINUTES', 20)
        self.RISK_OFF_MOMENTUM_THRESHOLD = kwargs.get('RISK_OFF_MOMENTUM_THRESHOLD', -0.015)
        self.RISK_OFF_ADX_THRESHOLD = kwargs.get('RISK_OFF_ADX_THRESHOLD', 35)
        self.RISK_OFF_CORRELATION_THRESHOLD = kwargs.get('RISK_OFF_CORRELATION_THRESHOLD', 0.90)
        self.RISK_OFF_DRAWDOWN_VELOCITY_ENABLED = kwargs.get('RISK_OFF_DRAWDOWN_VELOCITY_ENABLED', False)
        self.RISK_OFF_DRAWDOWN_VELOCITY = kwargs.get('RISK_OFF_DRAWDOWN_VELOCITY', 0.02)
        self.NORMAL_TOTAL_EXPOSURE_CAP = kwargs.get('NORMAL_TOTAL_EXPOSURE_CAP', 0.70)
        self.RISK_OFF_TOTAL_EXPOSURE_CAP = kwargs.get('RISK_OFF_TOTAL_EXPOSURE_CAP', 0.40)
        self.RECOVERY_STABILITY_BARS = kwargs.get('RECOVERY_STABILITY_BARS', 10)
        self.RECOVERY_MIN_TOTAL_BARS = kwargs.get('RECOVERY_MIN_TOTAL_BARS', 30)
        self.RECOVERY_ENTRY_MOMENTUM_MIN = kwargs.get('RECOVERY_ENTRY_MOMENTUM_MIN', -0.005)
        self.RECOVERY_ADVANCE_MOMENTUM_MIN = kwargs.get('RECOVERY_ADVANCE_MOMENTUM_MIN', 0.0)
        self.RECOVERY_POSITION_RAMP = kwargs.get('RECOVERY_POSITION_RAMP', [0.25, 0.5, 0.75, 1.0])
        self.RECOVERY_NO_NEW_LOW = kwargs.get('RECOVERY_NO_NEW_LOW', True)


def make_overlay(config=None, use_temp_state_file=True):
    """Create a RiskOverlay instance for testing."""
    if config is None:
        config = MockConfig()
    overlay = RiskOverlay(config)

    # Use temp file to avoid polluting real state
    if use_temp_state_file:
        import tempfile
        overlay.STATE_FILE = tempfile.mktemp(suffix='.json')
        overlay.COMMAND_FILE = tempfile.mktemp(suffix='.json')
        # Reset telemetry to avoid pollution from production state file
        # (RiskOverlay.__init__ loads state before we change STATE_FILE)
        overlay.telemetry = {
            "avoided_buys_count": 0,
            "avoided_buys_notional": 0.0,
            "cancelled_limits_count": 0,
            "cancelled_limits_notional": 0.0,
            "untracked_buys_count": 0,
            "rebalances_blocked_count": 0,
            "mode_transitions": [],
        }
        overlay.mode = RiskMode.NORMAL
        overlay.trigger_reasons = []

    return overlay


def make_crash_signals(momentum=-0.02, adx=40, adx_direction="down", correlation=0.95):
    """Create signals that would trigger RISK_OFF."""
    return {
        "momentum": momentum,
        "adx": adx,
        "adx_direction": adx_direction,
        "max_correlation": correlation,
        "current_prices": {"BTC/USD": 95000, "SOL/USD": 200},
    }


def make_normal_signals(momentum=0.005, adx=20, adx_direction="neutral", correlation=0.5):
    """Create signals for normal conditions."""
    return {
        "momentum": momentum,
        "adx": adx,
        "adx_direction": adx_direction,
        "max_correlation": correlation,
        "current_prices": {"BTC/USD": 96000, "SOL/USD": 210},
    }


class TestRiskOffTriggers:
    """Tests for RISK_OFF trigger conditions."""

    def test_single_signal_does_not_trigger(self):
        """Single signal alone should NOT trigger RISK_OFF (need 2-of-3)."""
        overlay = make_overlay()

        # Only momentum shock (1 of 3)
        signals = {
            "momentum": -0.02,  # Below threshold
            "adx": 25,  # Below threshold
            "adx_direction": "neutral",
            "max_correlation": 0.5,  # Below threshold
            "current_prices": {"BTC/USD": 95000},
        }

        result = overlay.evaluate(signals)
        assert result == RiskMode.NORMAL

    def test_two_signals_trigger_risk_off(self):
        """2 of 3 signals should trigger RISK_OFF."""
        overlay = make_overlay()

        # Momentum shock + correlation spike (2 of 3)
        signals = {
            "momentum": -0.02,  # Below threshold
            "adx": 25,  # Below ADX threshold
            "adx_direction": "neutral",
            "max_correlation": 0.92,  # Above threshold
            "current_prices": {"BTC/USD": 95000},
        }

        result = overlay.evaluate(signals)
        assert result == RiskMode.RISK_OFF

    def test_adx_only_triggers_with_direction_down(self):
        """ADX + direction must both be met (ADX > threshold AND direction == 'down')."""
        overlay = make_overlay()

        # High ADX but direction is UP (not a crash signal)
        signals = {
            "momentum": -0.02,  # 1 signal
            "adx": 40,  # Above threshold
            "adx_direction": "up",  # NOT down
            "max_correlation": 0.5,  # Below threshold
            "current_prices": {"BTC/USD": 95000},
        }

        result = overlay.evaluate(signals)
        # Only momentum counts (1 signal) - should NOT trigger RISK_OFF
        assert result == RiskMode.NORMAL

    def test_adx_down_counts_as_signal(self):
        """ADX above threshold + direction down should count as a signal."""
        overlay = make_overlay()

        # Momentum + ADX down (2 signals)
        signals = {
            "momentum": -0.02,  # Signal 1
            "adx": 40,  # Above threshold
            "adx_direction": "down",  # Signal 2
            "max_correlation": 0.5,  # Below threshold
            "current_prices": {"BTC/USD": 95000},
        }

        result = overlay.evaluate(signals)
        assert result == RiskMode.RISK_OFF

    def test_all_three_signals_trigger(self):
        """All 3 signals firing should definitely trigger RISK_OFF."""
        overlay = make_overlay()

        signals = make_crash_signals()
        result = overlay.evaluate(signals)

        assert result == RiskMode.RISK_OFF
        assert len(overlay.trigger_reasons) >= 2

    def test_drawdown_velocity_disabled_by_default(self):
        """Drawdown velocity should be disabled by default."""
        overlay = make_overlay()

        # Only drawdown velocity above threshold (should NOT count)
        signals = {
            "momentum": 0.0,
            "adx": 20,
            "adx_direction": "neutral",
            "max_correlation": 0.5,
            "drawdown_velocity": 0.05,  # High, but disabled
            "current_prices": {"BTC/USD": 95000},
        }

        result = overlay.evaluate(signals)
        assert result == RiskMode.NORMAL

    def test_drawdown_velocity_when_enabled(self):
        """Drawdown velocity should count when enabled."""
        config = MockConfig(RISK_OFF_DRAWDOWN_VELOCITY_ENABLED=True)
        overlay = make_overlay(config)

        # Momentum + drawdown velocity (2 signals when enabled)
        signals = {
            "momentum": -0.02,  # Signal 1
            "adx": 20,
            "adx_direction": "neutral",
            "max_correlation": 0.5,
            "drawdown_velocity": 0.05,  # Signal 2 (when enabled)
            "current_prices": {"BTC/USD": 95000},
        }

        result = overlay.evaluate(signals)
        assert result == RiskMode.RISK_OFF


class TestBuySellGating:
    """Tests for buy/sell gating in different modes."""

    def test_buys_allowed_in_normal(self):
        """Buys should be allowed in NORMAL mode."""
        overlay = make_overlay()
        overlay.mode = RiskMode.NORMAL

        allowed, reason = overlay.allows_buy("BTC/USD")
        assert allowed == True
        assert "normal" in reason.lower()

    def test_buys_blocked_in_risk_off(self):
        """Buys should be blocked in RISK_OFF mode."""
        overlay = make_overlay()
        overlay.mode = RiskMode.RISK_OFF
        overlay.trigger_reasons = ["test_trigger"]

        allowed, reason = overlay.allows_buy("BTC/USD", notional=1000)
        assert allowed == False
        assert "RISK_OFF" in reason

    def test_buys_allowed_in_recovery(self):
        """Buys should be allowed (with reduced size) in RECOVERY mode."""
        overlay = make_overlay()
        overlay.mode = RiskMode.RECOVERY

        allowed, reason = overlay.allows_buy("BTC/USD")
        assert allowed == True
        assert "recovery" in reason.lower()

    def test_sells_always_allowed(self):
        """Sells should be allowed in ALL modes."""
        overlay = make_overlay()

        # Test in NORMAL
        overlay.mode = RiskMode.NORMAL
        allowed, _ = overlay.allows_sell("BTC/USD")
        assert allowed == True

        # Test in RISK_OFF
        overlay.mode = RiskMode.RISK_OFF
        allowed, _ = overlay.allows_sell("BTC/USD")
        assert allowed == True

        # Test in RECOVERY
        overlay.mode = RiskMode.RECOVERY
        allowed, _ = overlay.allows_sell("BTC/USD")
        assert allowed == True


class TestRebalanceGating:
    """Tests for rebalance-down gating."""

    def test_rebalance_down_allowed_in_normal(self):
        """Rebalance-down should be allowed in NORMAL mode."""
        overlay = make_overlay()
        overlay.mode = RiskMode.NORMAL

        allowed, reason = overlay.allows_rebalance_down("BTC/USD")
        assert allowed == True
        assert "normal" in reason.lower()

    def test_rebalance_down_blocked_in_risk_off(self):
        """Rebalance-down should be blocked in RISK_OFF mode."""
        overlay = make_overlay()
        overlay.mode = RiskMode.RISK_OFF

        allowed, reason = overlay.allows_rebalance_down("BTC/USD")
        assert allowed == False
        assert "blocked" in reason.lower()

    def test_rebalance_down_blocked_in_recovery(self):
        """Rebalance-down should STILL be blocked in RECOVERY mode."""
        overlay = make_overlay()
        overlay.mode = RiskMode.RECOVERY

        allowed, reason = overlay.allows_rebalance_down("BTC/USD")
        assert allowed == False
        assert "blocked" in reason.lower()


class TestPositionMultiplier:
    """Tests for position size multiplier."""

    def test_multiplier_is_1_in_normal(self):
        """Position multiplier should be 1.0 in NORMAL mode."""
        overlay = make_overlay()
        overlay.mode = RiskMode.NORMAL

        mult = overlay.get_position_multiplier()
        assert mult == 1.0

    def test_multiplier_is_0_in_risk_off(self):
        """Position multiplier should be 0.0 in RISK_OFF mode (no buys)."""
        overlay = make_overlay()
        overlay.mode = RiskMode.RISK_OFF

        mult = overlay.get_position_multiplier()
        assert mult == 0.0

    def test_multiplier_ramps_in_recovery(self):
        """Position multiplier should follow ramp in RECOVERY mode."""
        overlay = make_overlay()
        overlay.mode = RiskMode.RECOVERY

        # Stage 0: 0.25x
        overlay.recovery_state.stage = 0
        assert overlay.get_position_multiplier() == 0.25

        # Stage 1: 0.5x
        overlay.recovery_state.stage = 1
        assert overlay.get_position_multiplier() == 0.5

        # Stage 2: 0.75x
        overlay.recovery_state.stage = 2
        assert overlay.get_position_multiplier() == 0.75

        # Stage 3: 1.0x
        overlay.recovery_state.stage = 3
        assert overlay.get_position_multiplier() == 1.0


class TestRecovery:
    """Tests for RECOVERY mode behavior."""

    def test_min_hold_time_before_recovery(self):
        """Must hold RISK_OFF for minimum time before entering RECOVERY."""
        config = MockConfig(RISK_OFF_MIN_HOLD_MINUTES=20)
        overlay = make_overlay(config)

        # Enter RISK_OFF
        signals = make_crash_signals()
        overlay.evaluate(signals)
        assert overlay.mode == RiskMode.RISK_OFF

        # Try to evaluate with normal signals immediately (still within hold time)
        normal_signals = make_normal_signals()
        overlay.evaluate(normal_signals)

        # Should still be in RISK_OFF (min hold not met)
        assert overlay.mode == RiskMode.RISK_OFF

    def test_recovery_after_min_hold(self):
        """Should enter RECOVERY after min hold + stability."""
        config = MockConfig(
            RISK_OFF_MIN_HOLD_MINUTES=0,  # No minimum hold for testing
            RECOVERY_STABILITY_BARS=2,
        )
        overlay = make_overlay(config)

        # Enter RISK_OFF
        overlay.mode = RiskMode.RISK_OFF
        overlay.mode_entered_at = datetime.now() - timedelta(minutes=30)  # Past min hold

        # Send enough stable signals to pass stability gate
        normal_signals = make_normal_signals()
        for _ in range(5):
            overlay.evaluate(normal_signals)

        # Should transition to RECOVERY
        assert overlay.mode == RiskMode.RECOVERY

    def test_relapse_returns_to_risk_off(self):
        """New trigger during RECOVERY should return to RISK_OFF."""
        overlay = make_overlay()
        overlay.mode = RiskMode.RECOVERY
        overlay.recovery_state = RecoveryState(stage=1, bars_in_stage=5)

        # Send crash signals
        crash_signals = make_crash_signals()
        result = overlay.evaluate(crash_signals)

        assert result == RiskMode.RISK_OFF
        assert overlay.recovery_state.stage == 0  # Reset


class TestTelemetry:
    """Tests for telemetry tracking."""

    def test_blocked_buy_recorded(self):
        """Blocked buys should be recorded in telemetry."""
        overlay = make_overlay()
        overlay.mode = RiskMode.RISK_OFF
        overlay.trigger_reasons = ["test"]

        initial_count = overlay.telemetry["avoided_buys_count"]
        initial_notional = overlay.telemetry["avoided_buys_notional"]

        # This should record a blocked buy
        overlay.allows_buy("BTC/USD", notional=1000)

        assert overlay.telemetry["avoided_buys_count"] == initial_count + 1
        assert overlay.telemetry["avoided_buys_notional"] == initial_notional + 1000

    def test_cancelled_limit_recorded(self):
        """Cancelled limits should be recorded in telemetry."""
        overlay = make_overlay()

        initial_count = overlay.telemetry["cancelled_limits_count"]
        initial_notional = overlay.telemetry["cancelled_limits_notional"]

        overlay.record_cancelled_limit("BTC/USD", 500)

        assert overlay.telemetry["cancelled_limits_count"] == initial_count + 1
        assert overlay.telemetry["cancelled_limits_notional"] == initial_notional + 500

    def test_rebalance_blocked_recorded(self):
        """Blocked rebalances should be recorded in telemetry."""
        overlay = make_overlay()
        overlay.mode = RiskMode.RISK_OFF

        initial_count = overlay.telemetry["rebalances_blocked_count"]

        overlay.allows_rebalance_down("BTC/USD")

        assert overlay.telemetry["rebalances_blocked_count"] == initial_count + 1

    def test_untracked_buy_recorded(self):
        """Untracked buys should be recorded in telemetry."""
        overlay = make_overlay()

        initial_count = overlay.telemetry["untracked_buys_count"]

        overlay.record_untracked_buy("BTC/USD", "order-123")

        assert overlay.telemetry["untracked_buys_count"] == initial_count + 1


class TestManualOverride:
    """Tests for manual override functionality."""

    def test_manual_override_to_risk_off(self):
        """Should be able to manually force RISK_OFF mode."""
        overlay = make_overlay()
        assert overlay.mode == RiskMode.NORMAL

        overlay.set_manual_override(RiskMode.RISK_OFF, "Manual test")

        assert overlay.manual_override == RiskMode.RISK_OFF

        # Evaluate should return override mode regardless of signals
        normal_signals = make_normal_signals()
        result = overlay.evaluate(normal_signals)

        assert result == RiskMode.RISK_OFF

    def test_manual_override_to_normal(self):
        """Should be able to manually force NORMAL mode."""
        overlay = make_overlay()
        overlay.mode = RiskMode.RISK_OFF
        overlay.trigger_reasons = ["real_trigger"]

        overlay.set_manual_override(RiskMode.NORMAL, "Manual test")

        # Evaluate should return override mode
        crash_signals = make_crash_signals()
        result = overlay.evaluate(crash_signals)

        assert result == RiskMode.NORMAL

    def test_clear_manual_override(self):
        """Should be able to clear manual override."""
        overlay = make_overlay()
        overlay.set_manual_override(RiskMode.RISK_OFF, "test")
        assert overlay.manual_override == RiskMode.RISK_OFF

        overlay.set_manual_override(None)

        assert overlay.manual_override is None

    def test_command_file_processing(self):
        """Should process command files from API."""
        overlay = make_overlay()

        # Write command file
        command = {
            "action": "set_override",
            "mode": "RISK_OFF",
            "reason": "API test",
        }
        with open(overlay.COMMAND_FILE, 'w') as f:
            json.dump(command, f)

        # Process command
        result = overlay.check_command_file()

        assert result == True
        assert overlay.manual_override == RiskMode.RISK_OFF

        # Command file should be deleted
        assert not os.path.exists(overlay.COMMAND_FILE)

    def test_command_file_clear_override(self):
        """Command file should be able to clear override."""
        overlay = make_overlay()
        overlay.set_manual_override(RiskMode.RISK_OFF, "test")

        # Write clear command
        command = {"action": "clear_override"}
        with open(overlay.COMMAND_FILE, 'w') as f:
            json.dump(command, f)

        result = overlay.check_command_file()

        assert result == True
        assert overlay.manual_override is None


class TestStatePersistence:
    """Tests for state persistence."""

    def test_state_saved_on_mode_change(self):
        """State should be saved when mode changes."""
        overlay = make_overlay()

        # Trigger RISK_OFF
        crash_signals = make_crash_signals()
        overlay.evaluate(crash_signals)

        assert overlay.mode == RiskMode.RISK_OFF
        assert os.path.exists(overlay.STATE_FILE)

        # Read state file
        with open(overlay.STATE_FILE) as f:
            state = json.load(f)

        assert state["mode"] == "RISK_OFF"

    def test_state_restored_on_init(self):
        """State should be restored on initialization."""
        overlay1 = make_overlay()

        # Trigger RISK_OFF
        crash_signals = make_crash_signals()
        overlay1.evaluate(crash_signals)
        state_file = overlay1.STATE_FILE

        # Create new overlay with same state file
        config = MockConfig()
        overlay2 = RiskOverlay(config)
        overlay2.STATE_FILE = state_file
        overlay2._load_state()

        assert overlay2.mode == RiskMode.RISK_OFF

    def test_telemetry_persisted(self):
        """Telemetry should survive save/load cycle."""
        overlay = make_overlay()
        overlay.mode = RiskMode.RISK_OFF
        overlay.trigger_reasons = ["test"]

        # Record some telemetry
        overlay.allows_buy("BTC/USD", notional=1000)
        overlay.record_cancelled_limit("BTC/USD", 500)

        overlay._save_state()
        state_file = overlay.STATE_FILE

        # Load into new overlay
        config = MockConfig()
        overlay2 = RiskOverlay(config)
        overlay2.STATE_FILE = state_file
        overlay2._load_state()

        assert overlay2.telemetry["avoided_buys_count"] == 1
        assert overlay2.telemetry["avoided_buys_notional"] == 1000
        assert overlay2.telemetry["cancelled_limits_count"] == 1
        assert overlay2.telemetry["cancelled_limits_notional"] == 500


class TestDisabled:
    """Tests for when risk overlay is disabled."""

    def test_disabled_returns_normal(self):
        """Disabled overlay should always return NORMAL."""
        config = MockConfig(RISK_OVERLAY_ENABLED=False)
        overlay = make_overlay(config)

        crash_signals = make_crash_signals()
        result = overlay.evaluate(crash_signals)

        assert result == RiskMode.NORMAL

    def test_disabled_allows_all_buys(self):
        """Disabled overlay should allow all buys."""
        config = MockConfig(RISK_OVERLAY_ENABLED=False)
        overlay = make_overlay(config)

        allowed, _ = overlay.allows_buy("BTC/USD")
        assert allowed == True

    def test_disabled_allows_rebalance_down(self):
        """Disabled overlay should allow rebalance-down."""
        config = MockConfig(RISK_OVERLAY_ENABLED=False)
        overlay = make_overlay(config)

        allowed, _ = overlay.allows_rebalance_down("BTC/USD")
        assert allowed == True

    def test_disabled_multiplier_is_1(self):
        """Disabled overlay should have multiplier of 1.0."""
        config = MockConfig(RISK_OVERLAY_ENABLED=False)
        overlay = make_overlay(config)

        mult = overlay.get_position_multiplier()
        assert mult == 1.0


class TestGetStatus:
    """Tests for get_status() method."""

    def test_status_contains_required_fields(self):
        """Status should contain all required fields."""
        overlay = make_overlay()
        status = overlay.get_status()

        assert "enabled" in status
        assert "mode" in status
        assert "mode_since" in status
        assert "mode_duration_minutes" in status
        assert "trigger_reasons" in status
        assert "position_multiplier" in status
        assert "telemetry" in status
        assert "manual_override" in status
        assert "caps" in status

    def test_status_recovery_state(self):
        """Status should include recovery state when in RECOVERY."""
        overlay = make_overlay()
        overlay.mode = RiskMode.RECOVERY
        overlay.recovery_state = RecoveryState(stage=2, bars_in_stage=5, total_bars_in_recovery=25)

        status = overlay.get_status()

        assert status["recovery"] is not None
        assert status["recovery"]["stage"] == 2
        assert status["recovery"]["bars_in_stage"] == 5
        assert status["recovery"]["total_bars_in_recovery"] == 25

    def test_status_no_recovery_in_normal(self):
        """Status should not include recovery state in NORMAL mode."""
        overlay = make_overlay()
        overlay.mode = RiskMode.NORMAL

        status = overlay.get_status()

        assert status["recovery"] is None


class TestModeTransitions:
    """Tests for mode transition recording."""

    def test_transition_recorded(self):
        """Mode transitions should be recorded."""
        overlay = make_overlay()

        # Trigger RISK_OFF
        crash_signals = make_crash_signals()
        overlay.evaluate(crash_signals)

        assert len(overlay.telemetry["mode_transitions"]) >= 1
        last_transition = overlay.telemetry["mode_transitions"][-1]
        assert last_transition["from"] == "NORMAL"
        assert last_transition["to"] == "RISK_OFF"

    def test_transitions_limited_to_50(self):
        """Mode transitions should be limited to last 50."""
        overlay = make_overlay()

        # Pre-fill with 100 transitions
        overlay.telemetry["mode_transitions"] = [
            {"timestamp": datetime.now().isoformat(), "from": "A", "to": "B", "reasons": []}
            for _ in range(100)
        ]

        # Record one more
        overlay._record_transition(RiskMode.NORMAL, RiskMode.RISK_OFF)

        # Should be capped at 50
        assert len(overlay.telemetry["mode_transitions"]) <= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
