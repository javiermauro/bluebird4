# Active Context — Current Focus

## Now
- **System health after crash/restart**: bot/dashboard/notifier were reported healthy post-restart; DB reconcile reported “synced” in the terminal transcript.
- **Primary focus**: operational resilience and incident forensics reliability.
- [2025-12-20 00:10] Operating model updated: this agent focuses on **advice + analysis**; Claude Code handles implementation. Primary objective remains **profitability**.
 - [2025-12-20 00:45] Observability improvements implemented by Claude Code (idempotent trades logging, positions_value snapshots, daily_summary recompute w/ Mac mini local date). Next step is runtime verification after restart.

## Recent Incident Notes
- [2025-12-20 00:00] Machine crash caused bot downtime (~23h). Existing Alpaca limit orders continued filling while offline; net equity change during downtime was positive (per terminal transcript).

## Risks / Gaps
- **Bot log persistence** is inconsistent depending on launch method:
  - `start.py` pipes output and may not persist logs to disk.
  - Shell `nohup ... > /tmp/bluebird-bot.log` persists logs, but `/tmp` can be cleared on reboot.
- [2025-12-20 00:30] **Observability gap** (from `data/bluebird.db`): `equity_snapshots` are populating, but `daily_summary` has 0 rows and `trades` only has data through 2025-12-11. Also `equity_snapshots.positions_value` appears `NULL` in recent data. This blocks accurate “why did P&L vanish?” diagnosis.

## Next Steps (Short List)
- [2025-12-20 00:00] Make bot logs durable and easy to audit after reboot (prefer file handler + rotation; choose non-ephemeral path if appropriate).
- [2025-12-20 00:00] Add/verify an auto-restart mechanism after reboot (macOS LaunchAgent/LaunchDaemon) so “crash → bot down” becomes “crash → auto-restart”.
- [2025-12-20 00:00] Standardize crash review checklist (status, reconcile, open orders/positions, overlay mode, grid status).
- [2025-12-20 00:20] Paper trading focus: defer “profit sweep to savings” mechanics until strategy shows stable positive expectancy; prioritize performance improvements + drawdown/giveback reduction first.
- [2025-12-20 00:35] Reporting preference: `daily_summary` should use **Mac mini local timezone** for day boundaries (operator-friendly), not UTC.
- [2025-12-20 01:05] AI exploration: current live trading loop is grid-first; if exploring AI, prefer **shadow-mode** first and restrict AI to **risk/regime gating** (reducing exposure / pausing buys) rather than short-horizon direction prediction.
- [2025-12-20 01:30] Decision: **Option 1** — treat `orders` as authoritative and do not backfill/remove `trades` right now. Focus remains on profitability + giveback reduction.


