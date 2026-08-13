#!/usr/bin/env bash
#
# NexusQuant — daily morning routine.
# Runs every weekday before the London open (installed in crontab as
# `0 6 * * 1-5`): freshens the local parquet data from the running MT5
# terminal, confirms freshness, then runs the live signal pass so every
# decision is made on fresh bars.
#
#   [1/3] python -m src.data.update --group full_fx   # freshen data
#   [2/3] python -m src.data.update --check           # confirm freshness
#   [3/3] python -m src.live.run --group full_fx      # decide on fresh bars
#
# Output: append-only log at logs/daily.log, auto-rotated to
# logs/daily.log.1 when it exceeds 2 MB (kept out of git either way).
# Note: the crontab line intentionally has NO `>> log 2>&1` redirect —
# the tee calls below already write the log; a redirect would double-write
# the banner lines. Do not "fix" that.
#
# Optional push-channel secrets (webhook / bot token) are sourced from
# $PROJECT/.env.live if present — gitignored, so never commit real values:
#
#   cp .env.example .env.live   # empty NEXUS_* placeholders
#   # then fill in:
#   NEXUS_DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
#   NEXUS_TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
#   NEXUS_TELEGRAM_CHAT_ID="123456789"
#
# Note: keep .env.live with LF line endings — CRLF (Windows-edited) appends
# a stray \r to values and silently breaks webhook auth.
#
set -u

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
PY="$PROJECT/venv/bin/python"
LOG_DIR="$PROJECT/logs"
LOG="$LOG_DIR/daily.log"

[ -x "$PY" ] || { echo "NexusQuant venv missing at $PY — run setup first" >&2; exit 1; }
mkdir -p "$LOG_DIR" || { echo "cannot create $LOG_DIR" >&2; exit 1; }

# --- rotate oversized log (keeps history bounded) -------------------------
if [ -f "$LOG" ] && [ "$(wc -c < "$LOG" 2>/dev/null || echo 0)" -gt 2097152 ]; then
  mv -f "$LOG" "$LOG.log.1"
fi

# --- optional secrets (cron environment has none of your exports) ---------
if [ -f "$PROJECT/.env.live" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$PROJECT/.env.live"
  set +a
fi

cd "$PROJECT" || exit 1

echo "" | tee -a "$LOG"
echo "===== $(date '+%F %T %Z') NexusQuant daily run =====" | tee -a "$LOG"

echo "--- [1/3] freshen data (incremental MT5 update) ---" | tee -a "$LOG"
"$PY" -m src.data.update --group full_fx >> "$LOG" 2>&1
UPDATE_RC=$?

echo "--- [2/3] freshness check ---" | tee -a "$LOG"
"$PY" -m src.data.update --check >> "$LOG" 2>&1
CHECK_RC=$?

echo "--- [3/3] live signal pass ---" | tee -a "$LOG"
"$PY" -m src.live.run --group full_fx >> "$LOG" 2>&1
LIVE_RC=$?

echo "--- done (update=$UPDATE_RC check=$CHECK_RC live=$LIVE_RC) ---" | tee -a "$LOG"

# Non-zero exit if any step failed, so a cron MAILTO (if configured) or a
# monitoring wrapper can detect it. Freshness/staleness findings themselves
# are reported inside the logs, not as a failure.
[ "$UPDATE_RC" -eq 0 ] && [ "$CHECK_RC" -eq 0 ] && [ "$LIVE_RC" -eq 0 ]
