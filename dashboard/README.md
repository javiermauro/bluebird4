# BLUEBIRD Dashboard

React dashboard for the BLUEBIRD trading bot, built with Vite.

## Quick Start

```bash
npm install
npm run dev      # Development server on :5173
npm run build    # Production build to dist/
```

## Environment Variables

The dashboard can be configured to connect to different backend instances using environment variables:

### Main Trading Bot

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_HOST` | `window.location.hostname` | Backend API hostname |
| `VITE_API_PORT` | `8000` | Backend API port |

### Training Dashboard

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_TRAINING_API_HOST` | `window.location.hostname` | Training API hostname |
| `VITE_TRAINING_API_PORT` | `8001` | Training API port |

### Examples

**Paper trading instance on port 8002:**
```bash
VITE_API_PORT=8002 npm run dev
```

**Remote backend:**
```bash
VITE_API_HOST=192.168.1.100 VITE_API_PORT=8000 npm run dev
```

**Build with custom config:**
```bash
VITE_API_PORT=8002 npm run build
```

## Validation

1. Run dashboard with no env vars → should work against `localhost:8000`
2. Run with `VITE_API_PORT=8002` → confirm all calls go to `:8002`
3. Check browser Network tab for any remaining `localhost:8000` calls

## Tech Stack

- React + Vite
- Chart.js for visualizations
- WebSocket for real-time updates
