---
name: block-env-commits
enabled: true
event: bash
pattern: git\s+(add|commit).*\.env
action: block
---

**SECURITY BLOCK: .env file detected in git command**

The `.env` file contains sensitive credentials:
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `TWILIO_*` credentials

**Never commit credentials to git.** They should remain in `.env` (which is gitignored).

If you need to share configuration, use `.env.example` with placeholder values.
