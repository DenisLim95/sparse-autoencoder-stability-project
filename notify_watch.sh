#!/usr/bin/env bash
#
# Watch a running topk_sweep_experiments.py job and push an alert when it dies, stalls,
# or logs something that needs a human.
#
# Attaches to an already-running job by reading its log file, so it can be started at any
# point during a run rather than only at launch.
#
#   ALERT_NTFY_TOPIC=my-sae-alerts-8f3k ./notify_watch.sh "$SAE_ROOT/sweep_3arm_1B.log"
#
# Transports, any combination (each is optional; with none configured it only prints):
#   ALERT_NTFY_TOPIC     ntfy.sh topic. Install the ntfy app, subscribe to the same topic.
#                        Pick something unguessable: the topic name IS the only access control.
#   ALERT_EMAIL          Forward alerts to email via ntfy (needs ALERT_NTFY_TOPIC too).
#                        Heavily rate-limited upstream, so it can drop under a burst.
#   ALERT_WEBHOOK_URL    Slack or Discord incoming webhook.
#   ALERT_HEARTBEAT_URL  Dead-man's-switch ping URL, e.g. healthchecks.io. See the note in
#                        heartbeat() for why this catches what the rest of the script cannot.
#
# Tunables: ALERT_STALL_MINUTES (default 20), ALERT_POLL_SECONDS (default 60).

set -uo pipefail

LOG_FILE="${1:-}"
if [[ -z "$LOG_FILE" ]]; then
    echo "usage: $0 <log-file> [pid]" >&2
    exit 2
fi

PATTERN="${ALERT_PROC_PATTERN:-topk_sweep_experiments}"
WATCH_PID="${2:-}"
STALL_MINUTES="${ALERT_STALL_MINUTES:-20}"
(( STALL_MINUTES < 1 )) && STALL_MINUTES=1   # 0 would fire on every poll
POLL_SECONDS="${ALERT_POLL_SECONDS:-60}"
HOST="$(hostname -s)"

notify() {
    local title="$1" body="$2" priority="${3:-default}"
    echo "[$(date '+%H:%M:%S')] ALERT ($title): ${body%%$'\n'*}"

    if [[ -n "${ALERT_NTFY_TOPIC:-}" ]]; then
        local -a hdr=(-H "Title: $title" -H "Priority: $priority")
        [[ -n "${ALERT_EMAIL:-}" ]] && hdr+=(-H "Email: $ALERT_EMAIL")
        curl -fsS --max-time 15 "${hdr[@]}" -d "$body" \
            "https://ntfy.sh/${ALERT_NTFY_TOPIC}" >/dev/null 2>&1 \
            || echo "  (ntfy push failed)"
    fi

    if [[ -n "${ALERT_WEBHOOK_URL:-}" ]]; then
        # Slack and Discord disagree on the field name and silently 400 on the wrong one.
        local field="text"
        [[ "$ALERT_WEBHOOK_URL" == *discord* ]] && field="content"
        local payload
        payload="$(printf '%s' "[$HOST] $title"$'\n'"$body" |
            python3 -c "import json,sys; print(json.dumps({'$field': sys.stdin.read()}))")"
        curl -fsS --max-time 15 -H 'Content-Type: application/json' \
            -d "$payload" "$ALERT_WEBHOOK_URL" >/dev/null 2>&1 \
            || echo "  (webhook post failed)"
    fi
}

# Everything else in this script runs ON the machine being watched, so it dies with the
# machine and stays silent exactly when a host failure or the 24h shutdown hits. An external
# service expecting a regular ping is the only thing that can alert on that, because the
# absence of a ping is the signal.
heartbeat() {
    [[ -n "${ALERT_HEARTBEAT_URL:-}" ]] || return 0
    curl -fsS --max-time 10 "$ALERT_HEARTBEAT_URL" >/dev/null 2>&1 || true
}

running() {
    if [[ -n "$WATCH_PID" ]]; then
        kill -0 "$WATCH_PID" 2>/dev/null
    else
        pgrep -f "$PATTERN" >/dev/null 2>&1
    fi
}

if ! running; then
    notify "run is not running" \
        "No process matching '$PATTERN' and nothing to watch. Start the run first." high
    exit 1
fi

# Resume from the current end of the log so a mid-run attach does not replay warnings that
# have already been seen and dealt with.
offset="$(wc -c < "$LOG_FILE" 2>/dev/null || echo 0)"
last_step_line="$(grep -a '^step ' "$LOG_FILE" 2>/dev/null | tail -n 1)"
last_progress_at="$(date +%s)"
stall_reported=0

notify "watching run" \
    "log: $LOG_FILE
progress: ${last_step_line:-<no step lines yet>}
alerting on: crash, ${STALL_MINUTES}m stall, upload failures, milestones" low

while true; do
    sleep "$POLL_SECONDS"
    heartbeat

    size="$(wc -c < "$LOG_FILE" 2>/dev/null || echo 0)"
    if (( size > offset )); then
        new="$(tail -c "+$((offset + 1))" "$LOG_FILE" 2>/dev/null)"
        offset="$size"

        step_line="$(printf '%s' "$new" | grep -a '^step ' | tail -n 1)"
        if [[ -n "$step_line" && "$step_line" != "$last_step_line" ]]; then
            last_step_line="$step_line"
            last_progress_at="$(date +%s)"
            if (( stall_reported )); then
                notify "run recovered" "Progress resumed: $step_line"
                stall_reported=0
            fi
        fi

        # A partial mirror means some checkpoints exist only on this machine's disk, which
        # is worth waking up for on an ephemeral box even though training continues.
        if partial="$(printf '%s' "$new" | grep -a 'only [0-9]* of' | tail -n 1)"; [[ -n "$partial" ]]; then
            notify "checkpoints did not reach the Hub" "$partial" high
        elif mirrored="$(printf '%s' "$new" | grep -a 'milestone reached' | tail -n 1)"; [[ -n "$mirrored" ]]; then
            notify "milestone banked" "$mirrored" low
        fi

        if fatal="$(printf '%s' "$new" | grep -aE 'Traceback|CUDA out of memory|SystemExit' | tail -n 1)"; [[ -n "$fatal" ]]; then
            notify "error in log" "$fatal" high
        fi
    fi

    if ! running; then
        # Distinguishes a clean finish from a crash by what the script printed last, since
        # a watcher that does not own the process cannot read its exit code.
        tail_lines="$(tail -n 25 "$LOG_FILE" 2>/dev/null)"
        if printf '%s' "$tail_lines" | grep -aq 'SWEEP COMPLETE\|sweep_summary'; then
            notify "run FINISHED" "$tail_lines"
        else
            notify "run DIED" "Process gone without a completion marker. Last 25 lines:

$tail_lines" urgent
        fi
        exit 0
    fi

    idle_minutes=$(( ($(date +%s) - last_progress_at) / 60 ))
    if (( idle_minutes >= STALL_MINUTES && !stall_reported )); then
        notify "run STALLED" \
            "Alive but no new step line for ${idle_minutes}m. Last progress:
$last_step_line

A hang usually means the dataset stream is blocked on network, not a GPU problem." high
        stall_reported=1
    fi
done
