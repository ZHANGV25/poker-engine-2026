#!/usr/bin/env bash
# Download river precomputed data from S3 and merge into submission
#
# Usage:
#   bash blueprint/merge_river_v2.sh           # Download + merge
#   bash blueprint/merge_river_v2.sh --check   # Just check progress

set -euo pipefail

S3_PATH="s3://poker-blueprint-2026/river_v2/"
LOCAL_DIR="/tmp/river_v2_download"
SUBMISSION_DIR="submission/data/river"
REGION="us-east-1"

if [[ "${1:-}" == "--check" ]]; then
    echo "=== S3 Progress ==="
    count=$(aws s3 ls "$S3_PATH" --region "$REGION" 2>&1 | grep -c "river_" || echo 0)
    echo "$count / 80730 boards uploaded"
    exit 0
fi

echo "=== Downloading from S3 ==="
mkdir -p "$LOCAL_DIR"
aws s3 sync "$S3_PATH" "$LOCAL_DIR" --region "$REGION" --exclude "status_*" --exclude "log_*"

count=$(ls "$LOCAL_DIR"/river_*.npz 2>/dev/null | wc -l | tr -d ' ')
echo "Downloaded: $count board files"

echo "=== Copying to submission ==="
mkdir -p "$SUBMISSION_DIR"
cp "$LOCAL_DIR"/river_*.npz "$SUBMISSION_DIR/"

echo "=== Stats ==="
du -sh "$SUBMISSION_DIR"
ls "$SUBMISSION_DIR" | wc -l
echo "files in $SUBMISSION_DIR"

echo "=== Done ==="
echo "Submission data ready. Upload the submission folder to competition."
