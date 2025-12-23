# Progress — Status & History

## Current Status
- [2025-12-20 00:00] Memory Bank initialized (core files created).
- [2025-12-20 00:10] Operating model: this agent provides advice/analysis; Claude Code performs coding/implementation. North Star remains profitability.

## Recent Work (High Signal)
- [2025-12-20 00:00] Crash/outage review captured from terminal transcript:
  - Downtime ~23 hours.
  - Existing resting limit orders filled while offline.
  - Post-restart health checks + DB reconcile indicated “synced”.
- [2025-12-20 00:20] Capital management idea logged: “profit sweep to savings” discussed; deferred until after paper performance is consistently profitable and variance is understood.
- [2025-12-20 00:45] Observability upgrade implemented (by Claude Code):
  - DB indexes: `idx_trades_unique`, `idx_daily_summary_date`
  - `record_trade()` made idempotent (side normalization, symbol-precision rounding, order_id guard, lazy config import)
  - Trade logging added to fast fill + startup reconciliation paths
  - `positions_value` added to equity snapshots (with fallback + throttled warning)
  - `recompute_daily_summary()` added (DB-derived, Mac mini local date grouping)
  - Startup/shutdown hooks added; daily_summary now has 1 row for today (per Claude report)
- [2025-12-20 01:05] AI discussion: AI modules exist in repo but appear not wired into the current grid-first live loop; decision is to explore AI only if it improves profitability/risk, starting in shadow mode and focusing on regime/risk gating rather than next-bar direction prediction.
- [2025-12-20 01:15] Post-restart verification: bot healthy; `positions_value` now recording in new `equity_snapshots`; `daily_summary` updating for today. `trades` still expected to remain stale until the next fill/reconcile event occurs.
- [2025-12-20 01:25] Observability follow-up: daily_summary now reflects today’s fills (trade counts/fees populated) and aligns with Alpaca history (example: 9 filled orders today). `trades` table may still remain legacy/stale depending on whether you chose to source daily rollups from `orders` vs `trades`.
- [2025-12-20 01:30] Decision: keep **Option 1** — `orders` is the authoritative fill ledger; no `trades` backfill/cleanup for now.

## Known Issues / Follow-ups
- **P1**: Bot log persistence depends on launch method; forensics can be incomplete after reboot if logs are only in `/tmp` or not written to disk.
- **P2**: Auto-restart after system reboot not guaranteed unless configured (e.g., LaunchAgent).
- **P1**: Verify in runtime: `trades` should start recording on next fills and `equity_snapshots.positions_value` should become non-NULL after restart. If either stays missing, treat as a P1 regression and investigate immediately.


