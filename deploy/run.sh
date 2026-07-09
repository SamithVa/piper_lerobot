#!/bin/bash
# One-command deploy. Usage: bash deploy/run.sh <preset> ["task"] [duration_s]
# e.g.: bash deploy/run.sh pi05 "Stack the cup on top of the bowl." 60
set -eu
cd "$(dirname "$0")/.."
ARGS=("${1:?usage: run.sh <preset> [task] [duration_s]}")
# note: if-statements, not `[ ... ] && ...` — a failing && list trips set -e
if [ -n "${2-}" ]; then ARGS+=(--task="$2"); fi
if [ -n "${3-}" ]; then ARGS+=(--duration_s="$3"); fi
exec python3 -m deploy.launch "${ARGS[@]}"
