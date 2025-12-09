#!/bin/bash
# BLUEBIRD 4.0 - Notifier Service Management Script
# Manages the SMS notification service via launchctl

SERVICE_NAME="com.bluebird.notifier"
PLIST_PATH="$HOME/Library/LaunchAgents/${SERVICE_NAME}.plist"
PID_FILE="/tmp/bluebird-notifier.pid"
LOG_FILE="/tmp/bluebird-notifier.log"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
GOLD='\033[0;33m'

print_header() {
    echo -e "${GOLD}"
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║       BLUEBIRD 4.0 - SMS Notifier Service            ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

get_status() {
    if launchctl list | grep -q "$SERVICE_NAME"; then
        if [ -f "$PID_FILE" ]; then
            PID=$(python3 -c "import json; print(json.load(open('$PID_FILE'))['pid'])" 2>/dev/null)
            if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
                echo "running"
                return 0
            fi
        fi
        echo "loaded"
        return 0
    else
        echo "stopped"
        return 1
    fi
}

start_service() {
    print_header
    echo -e "${BLUE}Starting notifier service...${NC}"

    if [ ! -f "$PLIST_PATH" ]; then
        echo -e "${RED}Error: LaunchAgent plist not found at $PLIST_PATH${NC}"
        echo "Please ensure the plist file exists."
        exit 1
    fi

    STATUS=$(get_status)
    if [ "$STATUS" = "running" ]; then
        echo -e "${YELLOW}Service is already running${NC}"
        exit 0
    fi

    launchctl load "$PLIST_PATH" 2>/dev/null
    launchctl start "$SERVICE_NAME" 2>/dev/null

    # Wait for service to start
    sleep 2

    STATUS=$(get_status)
    if [ "$STATUS" = "running" ] || [ "$STATUS" = "loaded" ]; then
        echo -e "${GREEN}Service started successfully${NC}"
        show_status
    else
        echo -e "${RED}Failed to start service. Check logs:${NC}"
        echo "  tail -f $LOG_FILE"
    fi
}

stop_service() {
    print_header
    echo -e "${BLUE}Stopping notifier service...${NC}"

    STATUS=$(get_status)
    if [ "$STATUS" = "stopped" ]; then
        echo -e "${YELLOW}Service is not running${NC}"
        exit 0
    fi

    launchctl stop "$SERVICE_NAME" 2>/dev/null
    launchctl unload "$PLIST_PATH" 2>/dev/null

    # Clean up PID file if it exists
    [ -f "$PID_FILE" ] && rm -f "$PID_FILE"

    echo -e "${GREEN}Service stopped${NC}"
}

restart_service() {
    print_header
    echo -e "${BLUE}Restarting notifier service...${NC}"
    stop_service
    sleep 1
    start_service
}

show_status() {
    print_header
    STATUS=$(get_status)

    case $STATUS in
        running)
            PID=$(python3 -c "import json; print(json.load(open('$PID_FILE'))['pid'])" 2>/dev/null)
            STARTED=$(python3 -c "import json; print(json.load(open('$PID_FILE'))['started_at'])" 2>/dev/null)
            echo -e "  Status: ${GREEN}Running${NC}"
            echo -e "  PID:    ${BLUE}$PID${NC}"
            echo -e "  Started: $STARTED"

            # Show SMS count if available
            if [ -f "/tmp/bluebird-notifier-count.json" ]; then
                COUNT=$(python3 -c "import json; print(json.load(open('/tmp/bluebird-notifier-count.json'))['count'])" 2>/dev/null)
                LAST=$(python3 -c "import json; print(json.load(open('/tmp/bluebird-notifier-count.json')).get('last_sent', 'Never'))" 2>/dev/null)
                echo -e "  SMS Today: ${GOLD}$COUNT${NC}"
                echo -e "  Last SMS: $LAST"
            fi
            ;;
        loaded)
            echo -e "  Status: ${YELLOW}Loaded (starting...)${NC}"
            ;;
        stopped)
            echo -e "  Status: ${RED}Stopped${NC}"
            ;;
    esac

    echo ""
    echo -e "  Log file: ${BLUE}$LOG_FILE${NC}"
}

show_logs() {
    print_header
    echo -e "${BLUE}Showing notifier logs (Ctrl+C to exit)...${NC}"
    echo ""

    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo -e "${YELLOW}No log file found at $LOG_FILE${NC}"
    fi
}

show_help() {
    print_header
    echo "Usage: $0 {start|stop|restart|status|logs}"
    echo ""
    echo "Commands:"
    echo "  start   - Start the notifier service"
    echo "  stop    - Stop the notifier service"
    echo "  restart - Restart the notifier service"
    echo "  status  - Show service status"
    echo "  logs    - Tail the service logs"
    echo ""
}

# Main command handler
case "$1" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        show_help
        exit 1
        ;;
esac

exit 0
