"""
Orchestrator - Thin Meta-Controller for Inventory Management

Consumes existing RiskOverlay + regime/time/momentum/correlation signals and adds:
1. Inventory episode tracking (how long we've been "stuck" holding inventory)
2. Staged LIMIT-sell liquidation (only in NORMAL overlay mode)
3. Mode-based gates and size multipliers

Critical Constraint: Orchestrator never overrides RiskOverlay decisions.
It can only add restrictions, not remove them.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.utils.atomic_io import atomic_write_json

# Project root for persistent state files (survives reboot, unlike /tmp)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_DIR = os.path.join(PROJECT_ROOT, "data", "state")

logger = logging.getLogger("Orchestrator")


class OrchestratorMode(Enum):
    """Orchestrator operating modes"""
    DEFENSIVE = "defensive"        # Buys blocked, sells allowed
    GRID_REDUCED = "grid_reduced"  # Reduced sizing (50%)
    GRID_FULL = "grid_full"        # Normal operation


@dataclass
class InventoryEpisode:
    """Tracks how long we've been holding meaningful inventory"""
    start_ts: datetime
    symbol: str
    start_inventory_pct: float


@dataclass
class OrchestratorContext:
    """Context passed from bot_grid.py to orchestrator"""
    # Overlay state
    overlay_mode: Any  # RiskMode enum
    overlay_allow_buy: bool
    overlay_allow_sell: bool
    overlay_position_mult: float
    overlay_reasons: List[str]

    # Regime state
    regime_allow_buy: bool
    regime_allow_sell: bool
    regime_size_mult: float
    regime: Any  # MarketRegime enum
    adx: float
    confidence: float

    # Time/momentum/correlation
    time_should_trade: bool
    time_quality: float
    momentum_allow_buy: bool
    corr_adjustment: float

    # Position state
    equity: float
    current_price: float
    position_qty: float
    avg_entry_price: Optional[float]
    unrealized_pnl_pct: Optional[float]

    # Symbol config
    investment_ratio: float  # From GRID_CONFIGS


@dataclass
class LiquidationDecision:
    """Decision to place a liquidation order"""
    enabled: bool
    reason: str  # "tp_trim", "loss_cut", "max_age"
    reduce_target_inventory_pct: float
    qty: float
    limit_price: float
    client_order_id: str  # ORCH-LIQ-{symbol}-{timestamp}


@dataclass
class OrchestratorDecision:
    """Result of orchestrator evaluation"""
    effective_mode: OrchestratorMode
    allow_buy: bool
    allow_sell: bool  # Always True. Sell gating owned by overlay/stop-loss/windfall; orchestrator never blocks sells.
    size_mult: float  # 0.0 to 1.0
    cancel_buy_limits: bool
    liquidation: Optional[LiquidationDecision]
    reasons: List[str]


class Orchestrator:
    """
    Thin meta-controller for inventory management.

    Respects RiskOverlay precedence and avoids duplicating existing gating/sizing logic.
    """

    STATE_FILE = os.path.join(STATE_DIR, "orchestrator.json")

    def __init__(self, config):
        self.config = config
        self.enabled = getattr(config, 'ORCHESTRATOR_ENABLED', False)
        self.enforce = getattr(config, 'ORCHESTRATOR_ENFORCE', False)
        self.liquidation_enabled = getattr(config, 'ORCHESTRATOR_LIQUIDATION_ENABLED', False)

        # Per-symbol state
        self.episodes: Dict[str, InventoryEpisode] = {}
        self.last_mode: Dict[str, OrchestratorMode] = {}
        self.last_mode_change_ts: Dict[str, datetime] = {}
        self.last_cancel_ts: Dict[str, datetime] = {}
        self.last_liq_ts: Dict[str, datetime] = {}
        self.last_liq_order_id: Dict[str, str] = {}

        # Manual per-symbol buy-block (v1: in-memory + persisted, no command-file)
        self.manual_blocks: Dict[str, Dict] = {}

        # Telemetry (mirrors RiskOverlay pattern)
        self.telemetry = {
            "buys_blocked_count": 0,
            "buys_blocked_notional": 0.0,
            "size_reductions_count": 0,
            "size_reduced_notional": 0.0,
            "cancels_issued_count": 0,
            "cancels_issued_notional": 0.0,
            "liquidations_placed_count": 0,
            "liquidations_placed_notional": 0.0,
            "mode_transitions": [],
        }

        # Track if state changed (for save-on-change)
        self._state_dirty = False

        self._load_state()

        # Pre-register all symbols from config so they appear in dashboard immediately
        symbols = getattr(config, 'SYMBOLS', [])
        for sym in symbols:
            if sym not in self.last_mode:
                self.last_mode[sym] = OrchestratorMode.GRID_FULL
                self._state_dirty = True
                logger.info(f"[ORCH] Pre-registered symbol: {sym}")

        # Save if any new symbols were pre-registered
        if self._state_dirty:
            self._save_state()
            self._state_dirty = False

        logger.info(f"Orchestrator initialized: enabled={self.enabled}, enforce={self.enforce}, "
                    f"liquidation={self.liquidation_enabled}")

    def evaluate(self, symbol: str, context: OrchestratorContext,
                 alpaca_open_orders: List = None) -> OrchestratorDecision:
        """
        Main entry point - produces an OrchestratorDecision.

        Called after overlay evaluation in handle_bar.
        alpaca_open_orders passed to check for existing liquidation orders.
        """
        reasons = []

        # OVERLAY CLAMP: If overlay is RISK_OFF or RECOVERY, force safe defaults
        # Orchestrator can still compute inventory/episode, but no actions.
        #
        # INTENT: Orchestrator returns "neutral outputs" (allow_buy=True, size_mult=1.0)
        # and relies on the outer combiner to apply RiskOverlay gates and sizing.
        # RiskOverlay gates and sizing dominate; orchestrator is passive in these modes.
        overlay_mode_value = getattr(context.overlay_mode, 'value', str(context.overlay_mode))
        if overlay_mode_value in ("RISK_OFF", "RECOVERY"):
            # Still track inventory for logging
            inventory_pct = self._calculate_inventory_pct(context)
            self._update_episode(symbol, inventory_pct)
            return OrchestratorDecision(
                effective_mode=self.last_mode.get(symbol, OrchestratorMode.GRID_FULL),
                allow_buy=True,  # Neutral - overlay handles blocking
                allow_sell=True,
                size_mult=1.0,   # Neutral - overlay handles sizing
                cancel_buy_limits=False,  # Overlay handles cancels in RISK_OFF
                liquidation=None,  # NEVER liquidate in RISK_OFF/RECOVERY
                reasons=[f"overlay_clamp:{overlay_mode_value}"]
            )

        # 1. Calculate inventory %
        inventory_pct = self._calculate_inventory_pct(context)

        # 2. Update inventory episode
        self._update_episode(symbol, inventory_pct)

        # 3. Select mode (with hysteresis + cooldown + cancel logic)
        new_mode, should_cancel = self._select_mode(symbol, context, inventory_pct)

        # 4. Record mode transition
        old_mode = self.last_mode.get(symbol, OrchestratorMode.GRID_FULL)
        if new_mode != old_mode:
            self.last_mode[symbol] = new_mode
            self.last_mode_change_ts[symbol] = datetime.now()
            self.telemetry["mode_transitions"].append({
                "ts": datetime.now().isoformat(),
                "symbol": symbol,
                "from": old_mode.value,
                "to": new_mode.value,
                "inventory_pct": inventory_pct
            })
            # Keep only last 100 transitions
            if len(self.telemetry["mode_transitions"]) > 100:
                self.telemetry["mode_transitions"] = self.telemetry["mode_transitions"][-100:]
            reasons.append(f"mode_change:{old_mode.value}->{new_mode.value}")
            self._state_dirty = True

        # 5. Compute gates and size based on mode
        if new_mode == OrchestratorMode.DEFENSIVE:
            allow_buy = False
            size_mult = 0.0
            reasons.append("DEFENSIVE:buys_blocked")
        elif new_mode == OrchestratorMode.GRID_REDUCED:
            allow_buy = True
            size_mult = 0.5
            reasons.append("GRID_REDUCED:50%_size")
        else:  # GRID_FULL
            allow_buy = True
            size_mult = 1.0

        # 6. Evaluate liquidation (only in NORMAL overlay mode)
        liq_decision = self._evaluate_liquidation(symbol, context, inventory_pct, alpaca_open_orders)
        if liq_decision:
            reasons.append(f"liquidation:{liq_decision.reason}")

        # 7. Persist state if changed
        if self._state_dirty:
            self._save_state()
            self._state_dirty = False

        return OrchestratorDecision(
            effective_mode=new_mode,
            allow_buy=allow_buy,
            allow_sell=True,  # Always allow sells
            size_mult=size_mult,
            cancel_buy_limits=should_cancel,
            liquidation=liq_decision,
            reasons=reasons
        )

    def _calculate_inventory_pct(self, context: OrchestratorContext) -> float:
        """
        Calculate inventory % = 100 * current_notional / target_notional

        target_notional = equity * investment_ratio
        current_notional = abs(position_qty) * current_price
        """
        target_notional = context.equity * context.investment_ratio
        current_notional = abs(context.position_qty) * context.current_price
        if target_notional <= 0:
            return 0.0
        return 100.0 * current_notional / target_notional

    def _update_episode(self, symbol: str, inventory_pct: float) -> Optional[InventoryEpisode]:
        """
        Track inventory episode lifecycle.

        Start episode when inventory >= EPISODE_START_PCT
        Reset episode when inventory <= EPISODE_RESET_PCT
        """
        EPISODE_START_PCT = getattr(self.config, 'EPISODE_START_PCT', 30)
        EPISODE_RESET_PCT = getattr(self.config, 'EPISODE_RESET_PCT', 10)

        episode = self.episodes.get(symbol)

        if episode is None and inventory_pct >= EPISODE_START_PCT:
            # Start new episode
            self.episodes[symbol] = InventoryEpisode(
                start_ts=datetime.now(),
                symbol=symbol,
                start_inventory_pct=inventory_pct
            )
            logger.info(f"[ORCH] Episode started for {symbol}: inventory={inventory_pct:.0f}%")
            self._state_dirty = True
        elif episode and inventory_pct <= EPISODE_RESET_PCT:
            # Reset episode
            del self.episodes[symbol]
            logger.info(f"[ORCH] Episode reset for {symbol}: inventory={inventory_pct:.0f}%")
            self._state_dirty = True

        return self.episodes.get(symbol)

    def _select_mode(self, symbol: str, context: OrchestratorContext,
                     inventory_pct: float) -> Tuple[OrchestratorMode, bool]:
        """
        Mode selection with hysteresis and cooldown.

        Returns (new_mode, should_cancel_buy_limits)
        """
        # Manual block overrides inventory-based logic
        if self.check_manual_block(symbol):
            return OrchestratorMode.DEFENSIVE, True  # Force DEFENSIVE + cancel

        # Config thresholds (all configurable for consistency)
        DEFENSIVE_ENTER = getattr(self.config, 'DEFENSIVE_INVENTORY_PCT', 150)
        DEFENSIVE_EXIT = getattr(self.config, 'DEFENSIVE_EXIT_PCT', 130)
        GRID_REDUCED_ENTER = getattr(self.config, 'GRID_REDUCED_ENTER_PCT', 100)
        GRID_REDUCED_EXIT = getattr(self.config, 'GRID_REDUCED_EXIT_PCT', 80)
        COOLDOWN_MINUTES = getattr(self.config, 'ORCHESTRATOR_COOLDOWN_MINUTES', 60)

        current_mode = self.last_mode.get(symbol, OrchestratorMode.GRID_FULL)
        last_change = self.last_mode_change_ts.get(symbol)
        should_cancel = False

        # Check cooldown (unless overlay escalates - not applicable here)
        if last_change:
            minutes_since = (datetime.now() - last_change).total_seconds() / 60
            if minutes_since < COOLDOWN_MINUTES:
                return current_mode, False  # Hold current mode

        # Mode selection with hysteresis
        if inventory_pct >= DEFENSIVE_ENTER:
            new_mode = OrchestratorMode.DEFENSIVE
        elif inventory_pct >= GRID_REDUCED_ENTER and current_mode != OrchestratorMode.DEFENSIVE:
            new_mode = OrchestratorMode.GRID_REDUCED
        elif inventory_pct < GRID_REDUCED_EXIT:
            new_mode = OrchestratorMode.GRID_FULL
        elif current_mode == OrchestratorMode.DEFENSIVE and inventory_pct < DEFENSIVE_EXIT:
            new_mode = OrchestratorMode.GRID_REDUCED  # Step down from DEFENSIVE
        else:
            new_mode = current_mode  # Maintain current mode in hysteresis band

        # Cancel buy limits ONLY if:
        # 1. Transitioning TO DEFENSIVE (not already there)
        # 2. Overlay is NORMAL (RiskOverlay handles cancels in RISK_OFF)
        # 3. Cancel cooldown passed
        overlay_mode_value = getattr(context.overlay_mode, 'value', str(context.overlay_mode))
        if (new_mode == OrchestratorMode.DEFENSIVE and
            current_mode != OrchestratorMode.DEFENSIVE and
            overlay_mode_value == "NORMAL"):
            last_cancel = self.last_cancel_ts.get(symbol)
            if not last_cancel or (datetime.now() - last_cancel).total_seconds() > 3600:
                should_cancel = True

        return new_mode, should_cancel

    def _evaluate_liquidation(self, symbol: str, context: OrchestratorContext,
                              inventory_pct: float,
                              alpaca_open_orders: List) -> Optional[LiquidationDecision]:
        """
        Check if liquidation is warranted (only in NORMAL overlay mode).
        Checks alpaca_open_orders for existing ORCH-LIQ orders.
        """
        # CRITICAL: Never liquidate in RISK_OFF or RECOVERY
        overlay_mode_value = getattr(context.overlay_mode, 'value', str(context.overlay_mode))
        if overlay_mode_value != "NORMAL":
            return None

        if not self.liquidation_enabled:
            return None

        # IDEMPOTENCY: Check for existing liquidation order
        if self._has_pending_liquidation_order(symbol, alpaca_open_orders):
            return None

        episode = self.episodes.get(symbol)
        if not episode:
            return None

        episode_age_hours = (datetime.now() - episode.start_ts).total_seconds() / 3600
        unrealized_pnl = context.unrealized_pnl_pct or 0

        # Config-driven thresholds
        TP_HOURS = getattr(self.config, 'LIQ_TP_HOURS', 24)
        TP_MIN_PNL = getattr(self.config, 'LIQ_TP_MIN_PNL_PCT', 0.003)
        TP_INV = getattr(self.config, 'LIQ_TP_INVENTORY_PCT', 120)
        TP_TARGET = getattr(self.config, 'LIQ_TP_TARGET_INV_PCT', 100)
        LOSS_HOURS = getattr(self.config, 'LIQ_LOSS_HOURS', 48)
        LOSS_CUT = getattr(self.config, 'LIQ_LOSS_CUT_PCT', -0.02)
        LOSS_INV = getattr(self.config, 'LIQ_LOSS_INVENTORY_PCT', 130)
        LOSS_REDUCE_MULT = getattr(self.config, 'LIQ_LOSS_REDUCE_MULT', 0.25)
        MAX_AGE_HOURS = getattr(self.config, 'LIQ_MAX_AGE_HOURS', 72)
        MAX_AGE_INV = getattr(self.config, 'LIQ_MAX_AGE_INVENTORY_PCT', 120)
        MIN_INTERVAL = getattr(self.config, 'LIQ_MIN_INTERVAL_MINUTES', 90) * 60  # seconds

        # Adaptive slippage based on ADX (stress proxy, not volatility)
        if context.adx > 30:
            SLIPPAGE = getattr(self.config, 'LIQ_SLIPPAGE_STRESSED', 0.008)
        else:
            SLIPPAGE = getattr(self.config, 'LIQ_SLIPPAGE_NORMAL', 0.004)

        # Rate limiting: one liquidation per symbol per MIN_INTERVAL
        if symbol in self.last_liq_ts:
            if (datetime.now() - self.last_liq_ts[symbol]).total_seconds() < MIN_INTERVAL:
                return None

        liq_decision = None
        client_order_id = f"ORCH-LIQ-{symbol.replace('/', '')}-{int(datetime.now().timestamp())}"

        # 1. TP Trim (take profit) - SAFEST
        if (episode_age_hours >= TP_HOURS and
            unrealized_pnl >= TP_MIN_PNL and
            inventory_pct >= TP_INV):
            liq_decision = LiquidationDecision(
                enabled=True,
                reason="tp_trim",
                reduce_target_inventory_pct=TP_TARGET,
                qty=self._calculate_reduction_qty(context, inventory_pct, TP_TARGET),
                limit_price=context.current_price * (1 - SLIPPAGE),
                client_order_id=client_order_id
            )

        # 2. Loss Cut (conservative) - Only reduce excess, not full position
        elif (episode_age_hours >= LOSS_HOURS and
              unrealized_pnl <= LOSS_CUT and
              inventory_pct >= LOSS_INV):
            excess_pct = inventory_pct - 100
            reduce_by_pct = excess_pct * LOSS_REDUCE_MULT  # 25%
            target = inventory_pct - reduce_by_pct
            liq_decision = LiquidationDecision(
                enabled=True,
                reason="loss_cut",
                reduce_target_inventory_pct=target,
                qty=self._calculate_reduction_qty(context, inventory_pct, target),
                limit_price=context.current_price * (1 - SLIPPAGE),
                client_order_id=client_order_id
            )

        # 3. Max Age Stop - Staged reduction
        elif (episode_age_hours >= MAX_AGE_HOURS and inventory_pct >= MAX_AGE_INV):
            liq_decision = LiquidationDecision(
                enabled=True,
                reason="max_age",
                reduce_target_inventory_pct=100,
                qty=self._calculate_reduction_qty(context, inventory_pct, 100),
                limit_price=context.current_price * (1 - SLIPPAGE),
                client_order_id=client_order_id
            )

        return liq_decision

    def _calculate_reduction_qty(self, context: OrchestratorContext,
                                  current_inv_pct: float, target_inv_pct: float) -> float:
        """
        Calculate qty to sell to reach target inventory %.
        Returns unrounded qty; executor applies round_qty().
        """
        if current_inv_pct <= target_inv_pct:
            return 0.0
        target_notional = context.equity * context.investment_ratio
        current_notional = abs(context.position_qty) * context.current_price
        target_position_notional = target_notional * (target_inv_pct / 100.0)
        reduction_notional = current_notional - target_position_notional
        if reduction_notional <= 0:
            return 0.0
        qty = reduction_notional / context.current_price
        # Cap at position qty
        qty = min(qty, abs(context.position_qty))
        return qty

    def _has_pending_liquidation_order(self, symbol: str, alpaca_open_orders: List) -> bool:
        """Check if there's already an open ORCH-LIQ order for this symbol"""
        prefix = f"ORCH-LIQ-{symbol.replace('/', '')}"
        for order in (alpaca_open_orders or []):
            client_id = getattr(order, 'client_order_id', '') or ''
            if client_id.startswith(prefix):
                return True
        return False

    def check_manual_block(self, symbol: str) -> bool:
        """Check if symbol is manually blocked (forces DEFENSIVE)"""
        return symbol in self.manual_blocks

    def set_manual_block(self, symbol: str, reason: str):
        """Manually block a symbol (forces DEFENSIVE mode)"""
        self.manual_blocks[symbol] = {
            "reason": reason,
            "blocked_at": datetime.now().isoformat()
        }
        logger.warning(f"[ORCH] Manual block SET for {symbol}: {reason}")
        self._save_state()

    def clear_manual_block(self, symbol: str):
        """Clear manual block for a symbol"""
        if symbol in self.manual_blocks:
            del self.manual_blocks[symbol]
            logger.info(f"[ORCH] Manual block CLEARED for {symbol}")
            self._save_state()

    def _record_blocked_buy(self, symbol: str, notional: float, reason: str):
        """Record telemetry for blocked buy"""
        self.telemetry["buys_blocked_count"] += 1
        self.telemetry["buys_blocked_notional"] += notional
        self._state_dirty = True

    def _save_state(self):
        """
        Persist to data/state/orchestrator.json.
        Uses atomic write (temp + rename) to prevent partial writes.
        """
        try:
            state = {
                "enabled": self.enabled,
                "enforce": self.enforce,
                "liquidation_enabled": self.liquidation_enabled,
                "symbols": {},
                "manual_blocks": self.manual_blocks,
                "telemetry": self.telemetry,
                "saved_at": datetime.now().isoformat()
            }

            # Build per-symbol state
            for symbol in set(list(self.last_mode.keys()) + list(self.episodes.keys())):
                episode = self.episodes.get(symbol)
                state["symbols"][symbol] = {
                    "mode": self.last_mode.get(symbol, OrchestratorMode.GRID_FULL).value,
                    "mode_changed_at": self.last_mode_change_ts.get(symbol, datetime.now()).isoformat()
                        if symbol in self.last_mode_change_ts else None,
                    "episode": {
                        "start_ts": episode.start_ts.isoformat(),
                        "start_inventory_pct": episode.start_inventory_pct
                    } if episode else None,
                    "last_cancel_ts": self.last_cancel_ts.get(symbol, datetime.now()).isoformat()
                        if symbol in self.last_cancel_ts else None,
                    "last_liq_ts": self.last_liq_ts.get(symbol, datetime.now()).isoformat()
                        if symbol in self.last_liq_ts else None,
                    "last_liq_order_id": self.last_liq_order_id.get(symbol)
                }

            # Use atomic write utility to prevent corruption on power loss
            atomic_write_json(self.STATE_FILE, state)

        except Exception as e:
            logger.error(f"[ORCH] Failed to save state: {e}")

    def _load_state(self):
        """Restore from state file on startup"""
        if not os.path.exists(self.STATE_FILE):
            return

        try:
            with open(self.STATE_FILE, 'r') as f:
                state = json.load(f)

            # Restore manual blocks
            self.manual_blocks = state.get("manual_blocks", {})

            # Restore telemetry
            self.telemetry = state.get("telemetry", self.telemetry)

            # Restore per-symbol state
            for symbol, sym_state in state.get("symbols", {}).items():
                # Restore mode
                mode_str = sym_state.get("mode", "grid_full")
                try:
                    self.last_mode[symbol] = OrchestratorMode(mode_str)
                except ValueError:
                    self.last_mode[symbol] = OrchestratorMode.GRID_FULL

                # Restore mode change timestamp
                if sym_state.get("mode_changed_at"):
                    try:
                        self.last_mode_change_ts[symbol] = datetime.fromisoformat(sym_state["mode_changed_at"])
                    except:
                        pass

                # Restore episode
                episode_data = sym_state.get("episode")
                if episode_data:
                    try:
                        self.episodes[symbol] = InventoryEpisode(
                            start_ts=datetime.fromisoformat(episode_data["start_ts"]),
                            symbol=symbol,
                            start_inventory_pct=episode_data["start_inventory_pct"]
                        )
                    except:
                        pass

                # Restore cancel/liq timestamps
                if sym_state.get("last_cancel_ts"):
                    try:
                        self.last_cancel_ts[symbol] = datetime.fromisoformat(sym_state["last_cancel_ts"])
                    except:
                        pass

                if sym_state.get("last_liq_ts"):
                    try:
                        self.last_liq_ts[symbol] = datetime.fromisoformat(sym_state["last_liq_ts"])
                    except:
                        pass

                if sym_state.get("last_liq_order_id"):
                    self.last_liq_order_id[symbol] = sym_state["last_liq_order_id"]

            logger.info(f"[ORCH] Loaded state: {len(self.last_mode)} symbols, "
                        f"{len(self.episodes)} active episodes")

        except Exception as e:
            logger.error(f"[ORCH] Failed to load state: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status for API"""
        return {
            "enabled": self.enabled,
            "enforce": self.enforce,
            "liquidation_enabled": self.liquidation_enabled,
            "symbols": {
                symbol: {
                    "mode": self.last_mode.get(symbol, OrchestratorMode.GRID_FULL).value,
                    "episode": {
                        "start_ts": self.episodes[symbol].start_ts.isoformat(),
                        "start_inventory_pct": self.episodes[symbol].start_inventory_pct,
                        "age_hours": (datetime.now() - self.episodes[symbol].start_ts).total_seconds() / 3600
                    } if symbol in self.episodes else None,
                    "manual_block": self.manual_blocks.get(symbol)
                }
                for symbol in set(list(self.last_mode.keys()) + list(self.episodes.keys()))
            },
            "telemetry": self.telemetry,
            "last_updated": datetime.now().isoformat()
        }
