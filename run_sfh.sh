#!/bin/bash

# Usage: ./run_sfh.sh <num_galaxies> <sfh_len>
LOGFILE="output.log"

python -u sfh.py "$1" "$2" --n_jobs 20 | tee "$LOGFILE" &

# Wait for the process to finish
wait

# Check for errors or warnings in the log file (case-insensitive)
if ! grep -i -E "error|warning" "$LOGFILE" > /dev/null; then
    rm "$LOGFILE"
fi