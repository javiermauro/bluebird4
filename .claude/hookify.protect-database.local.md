---
name: protect-database
enabled: true
event: all
pattern: (DROP\s+TABLE|DELETE\s+FROM\s+(trades|equity_snapshots|orders)|TRUNCATE|rm.*bluebird\.db)
action: block
---

**DATABASE PROTECTION: Destructive operation blocked**

The database `data/bluebird.db` contains critical trading history:

- **trades**: All executed trades with profit/loss calculations
- **equity_snapshots**: Hourly equity for drawdown tracking
- **orders**: Alpaca order records for reconciliation

**Losing this data means:**
- Cannot calculate true returns
- Circuit breakers lose drawdown history
- Reconciliation fails (fills not recovered)

If you need to reset, backup first:
```bash
cp data/bluebird.db data/bluebird.db.backup
```
