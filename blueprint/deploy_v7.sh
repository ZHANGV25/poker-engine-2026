#!/usr/bin/env bash
#
# Download v7.1 multi-street results from S3 and deploy to submission/data/.
#
# The fleet uploads per-board .npz files to s3://poker-blueprint-2026/multi_street_v7/.
# This script downloads them directly to submission/data/multi_street/ (no merge needed).
#
# Also downloads preflop v6 if available.
#
# Usage:
#   ./deploy_v7.sh                # Download and deploy
#   ./deploy_v7.sh --terminate    # Also terminate all instances
#   ./deploy_v7.sh --check        # Just check S3 counts, don't download

set -euo pipefail

AWS_PROFILE="default"
REGION="us-east-1"
S3_BUCKET="s3://poker-blueprint-2026"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOY_DIR="${PROJECT_DIR}/submission/data/multi_street"
PREFLOP_DIR="${PROJECT_DIR}/submission/data"

TERMINATE=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --terminate)  TERMINATE=true; shift ;;
        --check)      CHECK_ONLY=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--terminate] [--check]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ============================================================
# Check S3 for results
# ============================================================

log "Checking S3 for v7.1 multi-street results..."
BOARD_COUNT=$(aws s3 ls "${S3_BUCKET}/multi_street_v7/" --region "$REGION" 2>/dev/null | grep -c "board_" || echo 0)
INDEX_COUNT=$(aws s3 ls "${S3_BUCKET}/multi_street_v7/" --region "$REGION" 2>/dev/null | grep -c "index" || echo 0)
log "  Boards in S3: ${BOARD_COUNT} / 2925"

# Check preflop v6
PF_EXISTS=$(aws s3 ls "${S3_BUCKET}/results/preflop_strategy_v6.npz" --region "$REGION" 2>/dev/null | wc -l | tr -d ' ')
log "  Preflop v6: $([ "$PF_EXISTS" -gt 0 ] && echo 'available' || echo 'not yet')"

if $CHECK_ONLY; then
    # Also check fleet status
    log ""
    log "Fleet instance status:"
    aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=poker-blueprint-2026" "Name=instance-state-name,Values=running" \
        --query "Reservations[].Instances[].{ID:InstanceId,Type:InstanceType,Name:Tags[?Key=='Name']|[0].Value}" \
        --output table --region "$REGION" 2>/dev/null || echo "  No running instances"
    exit 0
fi

if [ "$BOARD_COUNT" -lt 2900 ]; then
    log "WARNING: Only ${BOARD_COUNT}/2925 boards available. Fleet may still be running."
    read -r -p "Deploy partial results? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        log "Aborted."
        exit 0
    fi
fi

# ============================================================
# Download multi-street results
# ============================================================

log "Downloading ${BOARD_COUNT} board files to ${DEPLOY_DIR}..."
mkdir -p "$DEPLOY_DIR"

# Back up existing data
if [ -d "$DEPLOY_DIR" ] && [ "$(ls "$DEPLOY_DIR"/board_*.npz 2>/dev/null | head -1)" ]; then
    BACKUP_DIR="${DEPLOY_DIR}.bak.$(date '+%Y%m%d_%H%M%S')"
    log "Backing up existing data to ${BACKUP_DIR}..."
    mv "$DEPLOY_DIR" "$BACKUP_DIR"
    mkdir -p "$DEPLOY_DIR"
fi

aws s3 sync "${S3_BUCKET}/multi_street_v7/" "$DEPLOY_DIR/" \
    --region "$REGION" --exclude "*.log"

DOWNLOADED=$(ls "$DEPLOY_DIR"/board_*.npz 2>/dev/null | wc -l | tr -d ' ')
log "Downloaded ${DOWNLOADED} board files."

# Verify a sample file
if [ "$DOWNLOADED" -gt 0 ]; then
    SAMPLE=$(ls "$DEPLOY_DIR"/board_*.npz | head -1)
    python3 -c "
import numpy as np
d = np.load('${SAMPLE}', allow_pickle=True)
print(f'  Sample: {\"${SAMPLE##*/}\"}')
print(f'    pot_sizes: {d[\"pot_sizes\"].tolist()}')
print(f'    flop_strategies: {d[\"flop_strategies\"].shape}')
print(f'    has opp strategies: {\"flop_opp_strategies\" in d}')
print(f'    has turn data: {\"turn_cards\" in d}')
if 'turn_cards' in d:
    print(f'    turn cards: {len(d[\"turn_cards\"])}')
" 2>/dev/null || log "  (could not verify sample file)"
fi

# ============================================================
# Download preflop v6 if available
# ============================================================

if [ "$PF_EXISTS" -gt 0 ]; then
    log ""
    log "Downloading preflop v6..."

    # Back up existing
    if [ -f "${PREFLOP_DIR}/preflop_strategy.npz" ]; then
        cp "${PREFLOP_DIR}/preflop_strategy.npz" "${PREFLOP_DIR}/preflop_strategy.npz.bak"
        cp "${PREFLOP_DIR}/preflop_tree.pkl" "${PREFLOP_DIR}/preflop_tree.pkl.bak" 2>/dev/null || true
    fi

    aws s3 cp "${S3_BUCKET}/results/preflop_strategy_v6.npz" "${PREFLOP_DIR}/preflop_strategy.npz" --region "$REGION"
    aws s3 cp "${S3_BUCKET}/results/preflop_tree_v6.pkl" "${PREFLOP_DIR}/preflop_tree.pkl" --region "$REGION" 2>/dev/null || true

    python3 -c "
import numpy as np
d = np.load('${PREFLOP_DIR}/preflop_strategy.npz')
print(f'  Preflop v6: {d[\"n_buckets\"]} buckets, {d[\"strategies\"].shape[0]} nodes')
print(f'  Raise levels: {d[\"raise_levels\"].tolist()}')
" 2>/dev/null || log "  (could not verify preflop file)"
else
    log ""
    log "Preflop v6 not available yet. Keeping v5."
fi

# ============================================================
# Terminate instances
# ============================================================

if $TERMINATE; then
    log ""
    log "Terminating all poker instances..."
    INSTANCE_IDS=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=poker-blueprint-2026" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query 'Reservations[].Instances[].InstanceId' \
        --output text --region "$REGION" 2>/dev/null || echo "")

    if [[ -n "$INSTANCE_IDS" ]]; then
        aws ec2 terminate-instances --instance-ids $INSTANCE_IDS --region "$REGION" --output text | head -5
        log "Terminated."
    else
        log "No instances to terminate."
    fi
fi

# ============================================================
# Summary
# ============================================================

log ""
log "=========================================="
log "Deploy Complete"
log "=========================================="
log "  Multi-street: ${DOWNLOADED} boards in ${DEPLOY_DIR}"
log "  Preflop: $([ "$PF_EXISTS" -gt 0 ] && echo 'v6 (200 buckets)' || echo 'v5 (50 buckets)')"
log ""
log "Next steps:"
log "  python test_path_coverage.py --quick    # Verify everything works"
log "  python run.py                           # Test match"
