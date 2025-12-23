# Project Brief — BLUEBIRD 4.0

## Summary
BLUEBIRD 4.0 is a cryptocurrency trading system focused on **grid trading** on Alpaca. It targets sideways/mean-reverting price action by maintaining buy/sell ladders (“grids”) across a configured price range, with multiple layers of safety controls (risk overlay, circuit breakers, sizing guardrails).

## Goals
- **Make money**: maximize long-run profitability (net of fees/slippage), with controlled drawdowns.
- **Consistent grid profit capture** in ranging markets (fee-aware).
- **Capital preservation** during crashes / downtrends via layered protections.
- **Operational resilience**: safe restarts, state reconciliation, single-instance protection.
- **Observability**: enough logging and persisted state to audit incidents and recover confidently.

## Non-Goals
- Pure price-direction prediction as the primary trading edge.
- High-frequency market-making or latency-sensitive strategies.
- Exchange-agnostic execution (system is Alpaca-first today).

## Primary Users
- Operator (you): runs services, monitors health/dashboard, reviews P&L and risk state.

## Success Criteria (Practical)
- Bot can restart cleanly after a crash/reboot and reconcile to Alpaca state without manual surgery.
- Risk overlay prevents “death spiral” buying during correlated selloffs/downtrends.
- Grid spacing and execution mode (maker vs taker) remains **net profitable after fees**.


