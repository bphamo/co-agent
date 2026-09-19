#!/usr/bin/env bash
# Stop hook: report the goal metric (strategy vs buy & hold, net of cost).
#
# Exits 0 on every path. A hook that can fail the session is worse than a hook
# that occasionally says nothing, and the two things it depends on are both
# routinely absent: the venv, and data/prices (gitignored, so a fresh clone has
# none). Silence in those cases is correct, not a fault to report.
set -uo pipefail

cd "$(dirname "$0")/../.." 2>/dev/null || exit 0

PY=.venv/bin/python
[ -x "$PY" ] || exit 0
[ -d data/prices ] || exit 0

out=$("$PY" -m co_agent.cycle.bottom_line --quiet-if-missing 2>&1) || exit 0
[ -n "$out" ] || exit 0

# systemMessage is what surfaces the reading in the UI.
"$PY" -c 'import json,sys; print(json.dumps({"systemMessage": sys.stdin.read().strip()}))' <<<"$out"
