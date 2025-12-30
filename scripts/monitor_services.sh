#!/bin/bash
# BLUEBIRD service monitor (macOS-friendly)
#
# Goals:
# - One line per component (no grouped sections)
# - Clear UP/RUNNING vs WAITING/DOWN status
# - Shows what is still missing when not READY
# - Exits automatically once READY unless --no-exit
#
# Usage:
#   bash scripts/monitor_services.sh
#   bash scripts/monitor_services.sh --no-exit
#   bash scripts/monitor_services.sh --interval 2
#   bash scripts/monitor_services.sh --once --no-clear

set -u

INTERVAL=2
NO_EXIT=0
ONCE=0
NO_CLEAR=0

while [ $# -gt 0 ]; do
  case "$1" in
    --interval)
      INTERVAL="${2:-2}"
      shift 2
      ;;
    --no-exit)
      NO_EXIT=1
      shift
      ;;
    --once)
      ONCE=1
      shift
      ;;
    --no-clear)
      NO_CLEAR=1
      shift
      ;;
    -h|--help)
      sed -n '1,160p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      echo "Run with --help for usage."
      exit 2
      ;;
  esac
done

BOT_URL="http://127.0.0.1:8000/health"
DASH_URL="http://127.0.0.1:5173"

launchd_state() {
  # Output: "<pid> <last_exit>" OR empty if not loaded
  local label="$1"
  launchctl list 2>/dev/null | awk -v l="$label" '$3==l {print $1" "$2}'
}

launchd_status_text() {
  local st="$1" # from launchd_state()
  if [ -z "$st" ]; then
    echo "NOT_LOADED"
    return 0
  fi
  local pid exitcode
  pid="$(echo "$st" | awk '{print $1}')"
  exitcode="$(echo "$st" | awk '{print $2}')"
  if [ "$pid" != "-" ]; then
    echo "RUNNING(pid=$pid)"
  else
    echo "LOADED(last_exit=$exitcode)"
  fi
}

check_http_ok() {
  local url="$1"
  curl -fsS --max-time 2 "$url" >/dev/null 2>&1
}

check_http_head_ok() {
  local url="$1"
  curl -fsSI --max-time 2 "$url" >/dev/null 2>&1
}

bot_listener_pid() {
  lsof -ti :8000 2>/dev/null | head -n 1 || true
}

compact_json() {
  python3 - <<'PY' 2>/dev/null
import json,sys
data=sys.stdin.read().strip()
try:
  obj=json.loads(data)
  print(json.dumps(obj, separators=(',',':')))
except Exception:
  print(data)
PY
}

print_row() {
  # name | status | detail
  local name="$1"; shift
  local status="$1"; shift
  local detail="$1"; shift || true
  printf "%-22s %-10s %s\n" "$name" "$status" "$detail"
}

while true; do
  if [ "$NO_CLEAR" -eq 0 ]; then
    clear
  fi
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "BLUEBIRD Monitor  |  ${now}  |  refresh ${INTERVAL}s"
  echo "===================================================================="
  printf "%-22s %-10s %s\n" "COMPONENT" "STATUS" "DETAIL"
  echo "--------------------------------------------------------------------"

  waiting=()

  # Bot API
  bot_ok="NO"
  bot_detail="url=${BOT_URL}"
  bot_json=""
  if check_http_ok "$BOT_URL"; then
    bot_ok="YES"
    bot_json="$(curl -fsS --max-time 2 "$BOT_URL" 2>/dev/null | head -c 800 || true)"
    if [ -n "$bot_json" ]; then
      bot_detail="$(echo "$bot_json" | compact_json)"
    fi
  else
    waiting+=("bot_api")
  fi
  print_row "Bot API (/health)" "$bot_ok" "$bot_detail"

  # Bot listener
  bl_pid="$(bot_listener_pid)"
  if [ -n "$bl_pid" ]; then
    print_row "Bot listener (:8000)" "YES" "pid=${bl_pid}"
  else
    print_row "Bot listener (:8000)" "NO" "no process bound"
    waiting+=("bot_listener")
  fi

  # Dashboard HTTP
  dash_ok="NO"
  dash_detail="url=${DASH_URL}"
  if check_http_head_ok "$DASH_URL"; then
    dash_ok="YES"
  else
    waiting+=("dashboard_http")
  fi
  print_row "Dashboard HTTP" "$dash_ok" "$dash_detail"

  # launchd jobs (each on its own line)
  for label in com.bluebird.dashboard com.bluebird.notifier com.bluebird.watchdog-bot com.bluebird.watchdog-notifier com.bluebird.monitor; do
    st="$(launchd_state "$label" || true)"
    txt="$(launchd_status_text "$st")"
    if [ "$txt" = "NOT_LOADED" ]; then
      print_row "$label" "NO" "$txt"
      waiting+=("$label")
    else
      print_row "$label" "YES" "$txt"
    fi
  done

  echo "--------------------------------------------------------------------"

  ready="YES"
  if [ "${#waiting[@]}" -gt 0 ]; then
    ready="NO"
  fi

  if [ "$ready" = "YES" ]; then
    echo "READY: YES"
  else
    echo "READY: NO  (waiting on: ${waiting[*]})"
  fi

  if [ "$ready" = "YES" ] && [ "$NO_EXIT" -eq 0 ]; then
    if [ "$ONCE" -eq 0 ]; then
      echo "All required components are up. Exiting."
    fi
    exit 0
  fi

  if [ "$ONCE" -eq 1 ]; then
    exit 0
  fi

  sleep "$INTERVAL"
done


