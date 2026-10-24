#!/bin/bash
# Use the provided message or default to a timestamped update
MSG="${1:-update: $(date '+%Y-%m-%d %H:%M:%S')}"

git add .
git commit -m "$MSG"
git push