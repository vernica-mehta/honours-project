#!/bin/bash

# Usage: ./run_sfh.sh <num_galaxies> <sfh_len>
LOGFILE="output.log"

nohup python -u sfh.py "$1" "$2" > "$LOGFILE" 2>&1

# Wait for the process to finish
wait

# Check for errors or warnings in the log file (case-insensitive)
if ! grep -i -E "error|warning" "$LOGFILE" > /dev/null; then
    rm "$LOGFILE"
fi