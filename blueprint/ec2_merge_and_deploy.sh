#!/usr/bin/env bash
#
# Download partial results from all fleet instances, merge them,
# and deploy to submission/data/.
#
# Run this after all instances have finished and uploaded their
# partial results to S3.
#
# Usage:
#   ./ec2_merge_and_deploy.sh                    # Download, merge, deploy
#   ./ec2_merge_and_deploy.sh --terminate        # Also terminate all instances
#   ./ec2_merge_and_deploy.sh --skip-download    # Merge from local cache
#   ./ec2_merge_and_deploy.sh --street flop      # Override street
#   ./ec2_merge_and_deploy.sh --dry-run          # Show what would happen

set -euo pipefail

# ============================================================
# Configuration
# ============================================================

AWS_PROFILE="default"
REGION="us-east-1"
S3_BUCKET="s3://poker-blueprint-2026"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOWNLOAD_DIR="${SCRIPT_DIR}/output/partial_results"
MERGED_DIR="${SCRIPT_DIR}/output"
DEPLOY_DIR="${PROJECT_DIR}/submission/data"

STREET="flop"
TERMINATE=false
SKIP_DOWNLOAD=false
DRY_RUN=false

# ============================================================
# Parse arguments
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --terminate)      TERMINATE=true; shift ;;
        --skip-download)  SKIP_DOWNLOAD=true; shift ;;
        --street)         STREET="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--terminate] [--skip-download] [--street flop|turn|river] [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================
# Helpers
# ============================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

separator() {
    echo "──────────────────────────────────────────────────────────"
}

# ============================================================
# Step 1: Verify all parts are available in S3
# ============================================================

log "Checking S3 for partial results..."
separator

PARTS_LIST=$(aws s3 ls "${S3_BUCKET}/unbucketed/" \
    --profile "$AWS_PROFILE" \
    --region "$REGION" 2>/dev/null || echo "")

if [[ -z "$PARTS_LIST" ]]; then
    echo "ERROR: No partial results found in ${S3_BUCKET}/unbucketed/"
    echo "Are the fleet instances still running? Check with: ./ec2_monitor.sh"
    exit 1
fi

# Count partial result files matching the expected pattern
PART_COUNT=0
echo "  Found partial results:"
while read -r date time size filename; do
    [[ -z "$filename" ]] && continue
    echo "    ${filename}  (${size} bytes, ${date} ${time})"
    PART_COUNT=$((PART_COUNT + 1))
done <<< "$PARTS_LIST"

echo ""
log "Found ${PART_COUNT} partial result files."

if $DRY_RUN; then
    log "[DRY RUN] Would download ${PART_COUNT} files to ${DOWNLOAD_DIR}"
    log "[DRY RUN] Would run merge_results.py"
    log "[DRY RUN] Would deploy to ${DEPLOY_DIR}"
    if $TERMINATE; then
        log "[DRY RUN] Would terminate fleet instances"
    fi
    exit 0
fi

# ============================================================
# Step 2: Download partial results
# ============================================================

if ! $SKIP_DOWNLOAD; then
    log "Downloading partial results to ${DOWNLOAD_DIR}..."
    mkdir -p "$DOWNLOAD_DIR"

    aws s3 sync "${S3_BUCKET}/unbucketed/" "$DOWNLOAD_DIR/" \
        --profile "$AWS_PROFILE" \
        --region "$REGION"

    DOWNLOADED=$(find "$DOWNLOAD_DIR" -type f -name "*.npz" | wc -l | tr -d ' ')
    log "Downloaded ${DOWNLOADED} files."

    # Also download logs for reference
    mkdir -p "${SCRIPT_DIR}/output/logs"
    aws s3 sync "${S3_BUCKET}/logs/" "${SCRIPT_DIR}/output/logs/" \
        --profile "$AWS_PROFILE" \
        --region "$REGION" 2>/dev/null || true
    log "Logs saved to ${SCRIPT_DIR}/output/logs/"
else
    log "Skipping download, using local files in ${DOWNLOAD_DIR}"
fi

echo ""

# ============================================================
# Step 3: Verify all parts are present
# ============================================================

log "Verifying partial results..."
separator

PART_FILES=$(find "$DOWNLOAD_DIR" -type f -name "${STREET}_part_*.npz" | sort)
PART_FILE_COUNT=$(echo "$PART_FILES" | grep -c . || echo 0)

if [[ "$PART_FILE_COUNT" -eq 0 ]]; then
    echo "ERROR: No ${STREET}_part_*.npz files found in ${DOWNLOAD_DIR}"
    echo "Files present:"
    ls -la "$DOWNLOAD_DIR/" 2>/dev/null || echo "  (directory empty)"
    exit 1
fi

log "Found ${PART_FILE_COUNT} partial files for street '${STREET}':"
echo "$PART_FILES" | while read -r f; do
    size=$(du -h "$f" | cut -f1)
    echo "  $(basename "$f")  ($size)"
done
echo ""

# ============================================================
# Step 4: Merge results
# ============================================================

log "Merging partial results..."
separator

MERGED_FILE="${MERGED_DIR}/${STREET}_blueprint.npz"

python3 "${SCRIPT_DIR}/merge_results.py" \
    --input-dir "$DOWNLOAD_DIR" \
    --output-file "$MERGED_FILE" \
    --street "$STREET"

MERGE_EXIT=$?
if [[ $MERGE_EXIT -ne 0 ]]; then
    echo "ERROR: merge_results.py failed with exit code ${MERGE_EXIT}"
    exit 1
fi

MERGED_SIZE=$(du -h "$MERGED_FILE" | cut -f1)
log "Merged blueprint: ${MERGED_FILE} (${MERGED_SIZE})"
echo ""

# ============================================================
# Step 5: Deploy to submission/data/
# ============================================================

log "Deploying to ${DEPLOY_DIR}..."
separator

mkdir -p "$DEPLOY_DIR"

# Back up existing blueprint if present
DEPLOY_TARGET="${DEPLOY_DIR}/${STREET}_blueprint.npz"
if [[ -f "$DEPLOY_TARGET" ]]; then
    BACKUP="${DEPLOY_TARGET}.bak.$(date '+%Y%m%d_%H%M%S')"
    cp "$DEPLOY_TARGET" "$BACKUP"
    log "Existing blueprint backed up to: $(basename "$BACKUP")"
fi

cp "$MERGED_FILE" "$DEPLOY_TARGET"
log "Deployed: ${DEPLOY_TARGET} (${MERGED_SIZE})"

# Also upload merged result to S3 for safekeeping
log "Uploading merged blueprint to S3..."
aws s3 cp "$MERGED_FILE" \
    "${S3_BUCKET}/merged/${STREET}_blueprint.npz" \
    --profile "$AWS_PROFILE" \
    --region "$REGION"
log "Uploaded to ${S3_BUCKET}/merged/${STREET}_blueprint.npz"
echo ""

# ============================================================
# Step 6: Optionally terminate fleet instances
# ============================================================

if $TERMINATE; then
    log "Terminating fleet instances..."
    separator

    INSTANCE_IDS=$(aws ec2 describe-instances \
        --filters \
            "Name=tag:Project,Values=poker-blueprint-2026" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query 'Reservations[].Instances[].InstanceId' \
        --output text \
        --profile "$AWS_PROFILE" \
        --region "$REGION" 2>/dev/null || echo "")

    if [[ -z "$INSTANCE_IDS" ]]; then
        log "No active instances to terminate."
    else
        INSTANCE_COUNT=$(echo "$INSTANCE_IDS" | wc -w | tr -d ' ')
        log "Terminating ${INSTANCE_COUNT} instances: ${INSTANCE_IDS}"

        aws ec2 terminate-instances \
            --instance-ids $INSTANCE_IDS \
            --profile "$AWS_PROFILE" \
            --region "$REGION" \
            --output text

        log "Termination initiated. Instances will shut down shortly."
    fi
    echo ""
fi

# ============================================================
# Summary
# ============================================================

echo ""
log "=========================================="
log "Merge and Deploy Complete"
log "=========================================="
log ""
log "  Street:     ${STREET}"
log "  Parts:      ${PART_FILE_COUNT} partial files merged"
log "  Blueprint:  ${DEPLOY_TARGET} (${MERGED_SIZE})"
log "  S3 backup:  ${S3_BUCKET}/merged/${STREET}_blueprint.npz"
if $TERMINATE; then
    log "  Instances:  terminated"
else
    log "  Instances:  still running (use --terminate to shut down)"
fi
log ""
log "To verify the blueprint:"
log "  python3 -c \"import numpy as np; d=np.load('${DEPLOY_TARGET}'); print({k: d[k].shape for k in d.files})\""
