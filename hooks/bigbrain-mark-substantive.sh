#!/bin/bash
# Runs after every tool call (Cursor `postToolUse` hook). Drops a per-conversation
# marker so the stop hook knows this turn did real work (used at least one tool).
# Turns that use no tools at all (pure conversational answers) leave no marker and
# are skipped by the maintenance pass. Always a no-op from the agent's perspective.
set -euo pipefail

input=$(cat)
cid=$(printf '%s' "$input" | jq -r '.conversation_id // "global"' | tr -c 'A-Za-z0-9._-' '_')

dir="${TMPDIR:-/tmp}/bigbrain-hooks"
mkdir -p "$dir"
: > "$dir/dirty-$cid"

printf '{}'
exit 0
