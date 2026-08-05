#!/bin/bash
# Cursor `stop` hook. Fires at the end of each agent turn. Injects a bigbrain
# memory-maintenance follow-up ONLY on the first stop after a real user message
# (loop_count == 0) AND only when a per-conversation "dirty" marker exists (the
# turn used >=1 tool, i.e. it did real work; pure conversational turns leave no
# marker and are skipped). On the follow-up turn's own stop (loop_count >= 1) it
# clears the marker and emits nothing, so the agent halts. This gates the pass to
# at most once per substantive user message and cannot loop.
set -euo pipefail

input=$(cat)
loop_count=$(printf '%s' "$input" | jq -r '.loop_count // 0')
cid=$(printf '%s' "$input" | jq -r '.conversation_id // "global"' | tr -c 'A-Za-z0-9._-' '_')

dir="${TMPDIR:-/tmp}/bigbrain-hooks"
marker="$dir/dirty-$cid"

# On the maintenance follow-up turn's stop, clear any marker it set and halt.
if [[ "$loop_count" -ne 0 ]]; then
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

jq -n '{
  followup_message: (
    "bigbrain memory-maintenance pass (auto-triggered). Review only what happened this turn. " +
    "If a durable, reusable learning emerged (infra gotcha + fix, convention/decision + rationale, " +
    "codebase/service map, or people/ownership), run the core loop from the bigbrain-memory rule: " +
    "(1) memory_recall the topic first; (2) if a related entry exists, update it in place (re-store the " +
    "same topic with on_conflict=replace \u2014 do NOT rely on memory_update by numeric id, and do not " +
    "create a near-duplicate); (3) only memory_store fresh if nothing related exists. " +
    "Skip secrets and one-off/transient details. If nothing durable emerged, reply exactly " +
    "\"No memory update needed.\" and stop. Do not start any new work."
  )
}'
exit 0
