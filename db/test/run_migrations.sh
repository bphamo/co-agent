#!/usr/bin/env bash
# Apply the migrations to a throwaway Postgres cluster and run db/test/asserts.sql.
#
# Nothing here touches a real database. The cluster is created in a temp
# directory and destroyed on exit.
#
#   ./db/test/run_migrations.sh
#
# Requires Postgres 16 server binaries. 0003_embeddings.sql needs pgvector and
# is skipped automatically when the extension is not installed.
set -euo pipefail

PGBIN="${PGBIN:-/usr/lib/postgresql/16/bin}"
PORT="${PORT:-55432}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d /tmp/co-agent-pgtest.XXXXXX)"
DATA="$WORK/data"
SOCK="$WORK/sock"

# Postgres refuses to run as root, so run the cluster as an unprivileged user.
RUNAS=""
if [ "$(id -u)" = "0" ]; then
    RUNAS="${PGTEST_USER:-pgtest}"
    id -u "$RUNAS" >/dev/null 2>&1 || useradd -M -s /usr/sbin/nologin "$RUNAS"
fi

cleanup() {
    if [ -d "$DATA" ]; then
        run "$PGBIN/pg_ctl -D '$DATA' -m immediate stop" >/dev/null 2>&1 || true
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

run() {  # run a command as the cluster owner
    if [ -n "$RUNAS" ]; then su -s /bin/bash "$RUNAS" -c "$1"; else bash -c "$1"; fi
}

mkdir -p "$DATA" "$SOCK"
chmod 755 "$WORK"
[ -n "$RUNAS" ] && chown -R "$RUNAS" "$WORK"

echo "==> initdb ($WORK)"
run "$PGBIN/initdb -D '$DATA' -U postgres -A trust" >/dev/null

echo "==> start"
run "$PGBIN/pg_ctl -D '$DATA' -l '$WORK/pg.log' -o \"-p $PORT -k '$SOCK' -h ''\" -w start" >/dev/null
PSQL=("psql" "-v" "ON_ERROR_STOP=1" "-h" "$SOCK" "-p" "$PORT" "-U" "postgres" "-d" "postgres" "-q")

"${PSQL[@]}" -c "CREATE DATABASE co_agent;" >/dev/null
PSQL=("psql" "-v" "ON_ERROR_STOP=1" "-h" "$SOCK" "-p" "$PORT" "-U" "postgres" "-d" "co_agent" "-q")

for m in "$REPO_ROOT"/db/migrations/*.sql; do
    name="$(basename "$m")"
    if [ "$name" = "0003_embeddings.sql" ]; then
        if ! "${PSQL[@]}" -tAc \
            "SELECT 1 FROM pg_available_extensions WHERE name='vector'" | grep -q 1; then
            echo "==> skip $name (pgvector not installed)"
            continue
        fi
    fi
    echo "==> apply $name"
    "${PSQL[@]}" -f "$m" >/dev/null
done

echo "==> asserts"
set +e
"${PSQL[@]}" -f "$REPO_ROOT/db/test/asserts.sql" >"$WORK/asserts.out" 2>&1
rc=$?
set -e
sed -e 's/^psql:[^ ]* //' -e 's/^NOTICE:  //' "$WORK/asserts.out" \
    | grep -E '^(ok |FAIL|ERROR)' || true

if [ "$rc" -ne 0 ]; then
    echo
    echo "one or more assertions failed:" >&2
    tail -n 30 "$WORK/asserts.out" >&2
    exit 1
fi
echo "==> all assertions held"
