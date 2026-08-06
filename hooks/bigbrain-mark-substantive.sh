#!/bin/bash
# PostToolUse hook (Cursor `postToolUse`, Claude Code `PostToolUse`). Drops a per-session
# marker so the stop hook knows this turn did real work (used at least one tool). Turns
# that use no tools at all leave no marker and are skipped by the maintenance pass.
# Always a no-op from the agent's perspective.
set -euo pipefail

# The maintenance pass drives its own headless agent session, which fires this same hook.
# That session's tool calls must not mark anything as needing a pass.
if [[ -n "${BIGBRAIN_MAINT:-}" ]]; then
  printf '{}'
  exit 0
fi

input=$(cat)
sid=$(printf '%s' "$input" | jq -r '.session_id // .conversation_id // "global"')
sid=$(printf '%s' "$sid" | tr -c 'A-Za-z0-9._-' '_')

dir="${TMPDIR:-/tmp}/bigbrain-hooks"
mkdir -p "$dir"
: > "$dir/dirty-$sid"

printf '{}'
exit 0
