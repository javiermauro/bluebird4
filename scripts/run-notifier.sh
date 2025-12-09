#!/bin/bash
# Wrapper script to run the notifier with proper environment
cd "/Volumes/DOCK/BLUEBIRD 4.0"

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Run the notifier with the project's python
exec /usr/bin/python3 src/notifications/notifier.py "$@"
