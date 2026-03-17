#!/usr/bin/env bash
#
# Monitor progress of blueprint computation fleet.
#
# Checks:
#   1. Instance states (running/stopped/terminated)
#   2. SSHs into each running instance to read status + tail logs
#   3. Lists completed partial results in S3
#
# Usage:
#   ./ec2_monitor.sh               # Full status report
#   ./ec2_monitor.sh --s3-only     # Only check S3 for completed parts
#   ./ec2_monitor.sh --ssh         # SSH into each and tail logs
#   ./ec2_monitor.sh --loop 60     # Re-check every 60 seconds
#   ./ec2_monitor.sh --instance 3  # Check only instance 3

set -euo pipefail

# ============================================================
# Configuration
# ============================================================

AWS_PROFILE="default"
REGION="us-east-1"
S3_BUCKET="s3://poker-blueprint-2026"
KEY_FILE="$HOME/.ssh/poker-debug-key.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR -i ${KEY_FILE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_LIST_FILE="${SCRIPT_DIR}/fleet_instances.txt"

MODE="full"        # full | s3-only | ssh
LOOP_INTERVAL=0    # 0 = run once
SINGLE_INSTANCE="" # empty = all instances

# ============================================================
# Parse arguments
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --s3-only)    MODE="s3-only"; shift ;;
        --ssh)        MODE="ssh"; shift ;;
        --loop)       LOOP_INTERVAL="$2"; shift 2 ;;
        --instance)   SINGLE_INSTANCE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--s3-only] [--ssh] [--loop SECONDS] [--instance N]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================
# Helpers
# ============================================================

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

separator() {
    echo "──────────────────────────────────────────────────────────"
}

# ============================================================
# Check instance states via AWS API
# ============================================================

check_instances() {
    log "Querying fleet instances..."
    separator

    # Find instances by project tag
    INSTANCES_JSON=$(aws ec2 describe-instances \
        --filters \
            "Name=tag:Project,Values=poker-blueprint-2026" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query 'Reservations[].Instances[].[
            InstanceId,
            State.Name,
            PublicIpAddress,
            LaunchTime,
            Tags[?Key==`Name`].Value | [0],
            Tags[?Key==`WorkerIndex`].Value | [0],
            Tags[?Key==`ClusterStart`].Value | [0],
            Tags[?Key==`ClusterEnd`].Value | [0]
        ]' \
        --output text \
        --profile "$AWS_PROFILE" \
        --region "$REGION" 2>/dev/null || echo "")

    if [[ -z "$INSTANCES_JSON" ]]; then
        echo "  No active fleet instances found."
        return
    fi

    RUNNING=0
    TOTAL=0

    printf "  %-4s %-20s %-10s %-16s %-15s %s\n" \
        "IDX" "INSTANCE_ID" "STATE" "IP" "CLUSTERS" "LAUNCHED"
    separator

    while IFS=$'\t' read -r iid state ip launch_time name worker_idx cluster_start cluster_end; do
        [[ -z "$iid" ]] && continue

        # Apply single-instance filter
        if [[ -n "$SINGLE_INSTANCE" && "$worker_idx" != "$SINGLE_INSTANCE" ]]; then
            continue
        fi

        TOTAL=$((TOTAL + 1))
        if [[ "$state" == "running" ]]; then
            RUNNING=$((RUNNING + 1))
        fi

        # Clean up None values
        [[ "$ip" == "None" ]] && ip="-"
        [[ "$worker_idx" == "None" ]] && worker_idx="?"
        [[ "$cluster_start" == "None" ]] && cluster_start="?"
        [[ "$cluster_end" == "None" ]] && cluster_end="?"

        local cluster_range="${cluster_start}-${cluster_end}"

        printf "  %-4s %-20s %-10s %-16s %-15s %s\n" \
            "$worker_idx" "$iid" "$state" "$ip" "$cluster_range" "$launch_time"
    done <<< "$INSTANCES_JSON"

    separator
    echo "  ${RUNNING}/${TOTAL} instances running"
    echo ""
}

# ============================================================
# SSH into instances and check status/logs
# ============================================================

check_instance_logs() {
    log "Checking instance logs via SSH..."
    separator

    # Build instance list from tags or from saved file
    local instance_data
    instance_data=$(aws ec2 describe-instances \
        --filters \
            "Name=tag:Project,Values=poker-blueprint-2026" \
            "Name=instance-state-name,Values=running" \
        --query 'Reservations[].Instances[].[
            PublicIpAddress,
            Tags[?Key==`WorkerIndex`].Value | [0],
            InstanceId
        ]' \
        --output text \
        --profile "$AWS_PROFILE" \
        --region "$REGION" 2>/dev/null || echo "")

    if [[ -z "$instance_data" ]]; then
        echo "  No running instances to check."
        return
    fi

    while IFS=$'\t' read -r ip worker_idx iid; do
        [[ -z "$ip" || "$ip" == "None" ]] && continue

        # Apply single-instance filter
        if [[ -n "$SINGLE_INSTANCE" && "$worker_idx" != "$SINGLE_INSTANCE" ]]; then
            continue
        fi

        echo ""
        echo "  Worker ${worker_idx} ($iid @ $ip)"
        echo "  ----------------------------------------"

        # Try to get status and last log lines
        STATUS=$(ssh $SSH_OPTS "ubuntu@${ip}" \
            "cat /opt/blueprint/status.txt 2>/dev/null || echo 'NO_STATUS'" 2>/dev/null || echo "SSH_FAILED")

        echo "  Status: ${STATUS}"

        if [[ "$STATUS" != "SSH_FAILED" ]]; then
            echo "  Last 5 log lines:"
            ssh $SSH_OPTS "ubuntu@${ip}" \
                "tail -5 /opt/blueprint/compute.log 2>/dev/null || tail -5 /var/log/blueprint-worker.log 2>/dev/null || echo '  (no log yet)'" \
                2>/dev/null | while read -r line; do echo "    $line"; done
        fi
    done <<< "$instance_data"

    echo ""
}

# ============================================================
# Check S3 for completed partial results
# ============================================================

check_s3_results() {
    log "Checking S3 for completed results..."
    separator

    echo "  Partial results in ${S3_BUCKET}/unbucketed/:"
    local parts
    parts=$(aws s3 ls "${S3_BUCKET}/unbucketed/" \
        --profile "$AWS_PROFILE" \
        --region "$REGION" 2>/dev/null || echo "")

    if [[ -z "$parts" ]]; then
        echo "    (none yet)"
    else
        local count=0
        local total_size=0
        while read -r date time size filename; do
            [[ -z "$filename" ]] && continue
            count=$((count + 1))
            total_size=$((total_size + size))
            printf "    %-40s %8s bytes  %s %s\n" "$filename" "$size" "$date" "$time"
        done <<< "$parts"
        echo ""
        echo "  Total: ${count} files, $(( total_size / 1024 / 1024 )) MB"
    fi

    echo ""

    echo "  Worker logs in ${S3_BUCKET}/logs/:"
    aws s3 ls "${S3_BUCKET}/logs/" \
        --profile "$AWS_PROFILE" \
        --region "$REGION" 2>/dev/null \
        | while read -r date time size filename; do
            printf "    %-40s %s %s\n" "$filename" "$date" "$time"
        done || echo "    (none yet)"

    echo ""
}

# ============================================================
# Main
# ============================================================

run_check() {
    echo ""
    echo "=========================================="
    echo "Blueprint Fleet Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    case "$MODE" in
        full)
            check_instances
            check_instance_logs
            check_s3_results
            ;;
        s3-only)
            check_s3_results
            ;;
        ssh)
            check_instance_logs
            ;;
    esac
}

if (( LOOP_INTERVAL > 0 )); then
    log "Monitoring every ${LOOP_INTERVAL}s. Press Ctrl+C to stop."
    while true; do
        run_check
        sleep "$LOOP_INTERVAL"
    done
else
    run_check
fi
