#!/usr/bin/env bash
#
# Package the blueprint and submission code and upload to S3.
#
# Creates a tar.gz of the blueprint/ and submission/ directories,
# then uploads to s3://poker-blueprint-2026/code/poker-blueprint.tar.gz.
#
# This must be run before ec2_launch_fleet.sh so instances can download
# the code during boot.
#
# Usage:
#   ./package_code.sh              # Package and upload
#   ./package_code.sh --dry-run    # Show what would be packaged
#   ./package_code.sh --local-only # Create tarball but don't upload

set -euo pipefail

# ============================================================
# Configuration
# ============================================================

AWS_PROFILE="default"
REGION="us-east-1"
S3_BUCKET="s3://poker-blueprint-2026"
S3_KEY="code/poker-blueprint.tar.gz"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TARBALL="/tmp/poker-blueprint.tar.gz"

DRY_RUN=false
LOCAL_ONLY=false

# ============================================================
# Parse arguments
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)    DRY_RUN=true; shift ;;
        --local-only) LOCAL_ONLY=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--local-only]"
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

# ============================================================
# Main
# ============================================================

log "Packaging code from: ${PROJECT_DIR}"

# Verify expected directories exist
if [[ ! -d "${PROJECT_DIR}/blueprint" ]]; then
    echo "ERROR: blueprint/ directory not found in ${PROJECT_DIR}"
    exit 1
fi
if [[ ! -d "${PROJECT_DIR}/submission" ]]; then
    echo "ERROR: submission/ directory not found in ${PROJECT_DIR}"
    exit 1
fi

# Show what will be included
log "Contents to package:"
echo "  blueprint/"
for f in "${PROJECT_DIR}"/blueprint/*.py; do
    [[ -f "$f" ]] && echo "    $(basename "$f")"
done
echo "  submission/"
for f in "${PROJECT_DIR}"/submission/*.py; do
    [[ -f "$f" ]] && echo "    $(basename "$f")"
done
if [[ -d "${PROJECT_DIR}/submission/data" ]]; then
    echo "  submission/data/"
    data_count=$(find "${PROJECT_DIR}/submission/data" -type f | wc -l | tr -d ' ')
    echo "    (${data_count} files)"
fi

if $DRY_RUN; then
    log "[DRY RUN] Would create: ${TARBALL}"
    log "[DRY RUN] Would upload to: ${S3_BUCKET}/${S3_KEY}"
    exit 0
fi

# Remove old tarball if it exists
rm -f "$TARBALL"

# Create tarball from project root
# Exclude __pycache__, .pyc, output dirs, logs, and the tarball scripts themselves
log "Creating tarball..."
tar czf "$TARBALL" \
    -C "$PROJECT_DIR" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='blueprint/output' \
    --exclude='blueprint/fleet_instances.txt' \
    --exclude='agent_logs' \
    --exclude='.git' \
    blueprint/ \
    submission/

TARBALL_SIZE=$(du -h "$TARBALL" | cut -f1)
log "Tarball created: ${TARBALL} (${TARBALL_SIZE})"

# List tarball contents for verification
log "Tarball contents (top-level):"
tar tzf "$TARBALL" | head -30 | while read -r line; do
    echo "  $line"
done
TOTAL_FILES=$(tar tzf "$TARBALL" | wc -l | tr -d ' ')
echo "  ... (${TOTAL_FILES} files total)"

if $LOCAL_ONLY; then
    log "Local-only mode. Tarball at: ${TARBALL}"
    exit 0
fi

# Upload to S3
log "Uploading to ${S3_BUCKET}/${S3_KEY}..."
aws s3 cp "$TARBALL" "${S3_BUCKET}/${S3_KEY}" \
    --profile "$AWS_PROFILE" \
    --region "$REGION"

# Verify upload
REMOTE_SIZE=$(aws s3 ls "${S3_BUCKET}/${S3_KEY}" \
    --profile "$AWS_PROFILE" \
    --region "$REGION" \
    | awk '{print $3}')

log "Upload complete."
log "  Local:  ${TARBALL} (${TARBALL_SIZE})"
log "  Remote: ${S3_BUCKET}/${S3_KEY} (${REMOTE_SIZE} bytes)"
log ""
log "Ready to launch fleet: ./ec2_launch_fleet.sh"
