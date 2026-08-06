#!/bin/bash
# Stop hook (Cursor `stop`, Claude Code `Stop`). Hands the bigbrain memory-maintenance
# pass to a detached background worker rather than continuing the turn, so the pass never
# appears in the session transcript the user is reading.
#
# The pass runs only on the first stop after a real user message and only when a
# per-session "dirty" marker exists, meaning the turn used at least one tool. Pure
# conversational turns leave no marker and are skipped.
set -euo pipefail

# The worker drives its own headless agent session, which fires this same hook. Without
# this guard every pass would spawn another one.
if [[ -n "${BIGBRAIN_MAINT:-}" ]]; then
  printf '{}'
  exit 0
fi

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

input=$(cat)
sid=$(printf '%s' "$input" | jq -r '.session_id // .conversation_id // "global"')
sid=$(printf '%s' "$sid" | tr -c 'A-Za-z0-9._-' '_')

# Claude Code reports an in-progress hook continuation as stop_hook_active; Cursor counts
# them in loop_count. Either way a non-zero value means this stop is not a fresh turn.
active=$(printf '%s' "$input" | jq -r '
  if (.stop_hook_active // false) then "true"
  elif ((.loop_count // 0) > 0) then "true"
  else "false" end')

dir="${TMPDIR:-/tmp}/bigbrain-hooks"
marker="$dir/dirty-$sid"

if [[ "$active" == "true" ]]; then
  rm -f "$marker"
  printf '{}'
  exit 0
fi

# Real user turn just ended: only run the pass if the turn did real work.
if [[ ! -f "$marker" ]]; then
  printf '{}'
  exit 0
fi
rm -f "$marker"

mkdir -p "$dir"
payload=$(mktemp "$dir/payload-$sid.XXXXXX")
printf '%s' "$input" > "$payload"

# Neither host has a non-blocking hook mode that also survives the session ending, so the
# worker is detached and this hook returns immediately.
if command -v setsid >/dev/null 2>&1; then
  setsid "$here/bigbrain-maintenance-run.sh" "$payload" </dev/null >/dev/null 2>&1 &
else
  # macOS ships no setsid. Double-fork instead: the intermediate shell exits at once and
  # the worker is reparented, so it outlives this hook and the session that spawned it.
  ( nohup "$here/bigbrain-maintenance-run.sh" "$payload" </dev/null >/dev/null 2>&1 & ) &
fi

printf '{}'
exit 0
