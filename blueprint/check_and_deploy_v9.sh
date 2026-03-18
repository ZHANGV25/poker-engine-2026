#!/usr/bin/env bash
# Check if EC2 v9 compute is done and deploy if ready.
# Run this in the morning: bash blueprint/check_and_deploy_v9.sh

set -euo pipefail
REGION="us-east-1"
S3_BUCKET="s3://poker-blueprint-2026"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Check S3 for results
BOARD_COUNT=$(aws s3 ls "${S3_BUCKET}/multi_street_v9/" --region "$REGION" 2>/dev/null | grep -c "board_" || echo 0)
log "Boards on S3: ${BOARD_COUNT} / 2925"

# Check running instances
RUNNING=$(aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=poker-blueprint-2026" "Name=instance-state-name,Values=running" \
    --query "Reservations[].Instances[].InstanceId" \
    --output text --region "$REGION" 2>/dev/null | wc -w | tr -d ' ')
log "Running instances: ${RUNNING}"

if [ "$BOARD_COUNT" -lt 2900 ]; then
    log "Not done yet. ${BOARD_COUNT}/2925 boards. ${RUNNING} instances still running."
    log "Check back later or re-run this script."
    exit 0
fi

log "✓ All boards ready! Downloading and compressing..."

# Download all board files
DEPLOY_DIR="submission/data/multi_street_v9"
mkdir -p "$DEPLOY_DIR"
aws s3 sync "${S3_BUCKET}/multi_street_v9/" "$DEPLOY_DIR/" \
    --region "$REGION" --exclude "*.log"

DOWNLOADED=$(ls "$DEPLOY_DIR"/board_*.npz 2>/dev/null | wc -l | tr -d ' ')
log "Downloaded ${DOWNLOADED} board files."

# Verify a sample
python3 -c "
import numpy as np
import os, sys
d = np.load('${DEPLOY_DIR}/board_0.npz', allow_pickle=True)
print(f'Sample board_0:')
print(f'  pot_sizes: {d[\"pot_sizes\"].tolist()}')
print(f'  flop_strategies: {d[\"flop_strategies\"].shape}')
print(f'  has opp: {\"flop_opp_strategies\" in d}')
print(f'  has turn: {\"turn_cards\" in d}')
n_pots = d['flop_strategies'].shape[0]
if n_pots != 5:
    print(f'  *** WARNING: expected 5 pots, got {n_pots} ***')
    sys.exit(1)
print(f'  ✓ 5 pots confirmed')
"

log ""
log "Next step: merge into LZMA and deploy."
log "Run: python3 blueprint/merge_v9.py"
log ""

# Terminate any remaining instances
if [ "$RUNNING" -gt 0 ]; then
    log "Terminating remaining instances..."
    INSTANCE_IDS=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=poker-blueprint-2026" "Name=instance-state-name,Values=running" \
        --query 'Reservations[].Instances[].InstanceId' \
        --output text --region "$REGION" 2>/dev/null || echo "")
    if [ -n "$INSTANCE_IDS" ]; then
        aws ec2 terminate-instances --instance-ids $INSTANCE_IDS --region "$REGION" --output text | head -3
    fi
fi

log "Done."
