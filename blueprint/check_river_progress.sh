#!/usr/bin/env bash
# Quick progress check for river precompute fleet.
#
# Usage:
#   bash blueprint/check_river_progress.sh          # S3 count + instance status
#   bash blueprint/check_river_progress.sh --ssh     # Also SSH into running instances

set -euo pipefail

REGION="us-east-1"
S3_PATH="s3://poker-blueprint-2026/river_v2/"
TOTAL_BOARDS=80730

echo "=== S3 Progress ==="
count=$(aws s3 ls "$S3_PATH" --region "$REGION" 2>&1 | grep -c "river_" || echo 0)
pct=$((count * 100 / TOTAL_BOARDS))
echo "$count / $TOTAL_BOARDS boards uploaded ($pct%)"
echo

echo "=== Running Instances ==="
aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Project,Values=poker-2026" "Name=instance-state-name,Values=running" \
    --query 'Reservations[*].Instances[*].[Tags[?Key==`Name`].Value|[0],InstanceId,PublicIpAddress,LaunchTime]' \
    --output table 2>&1

echo
echo "=== Status Files ==="
aws s3 ls "${S3_PATH}" --region "$REGION" 2>&1 | grep "status_" | while read -r line; do
    fname=$(echo "$line" | awk '{print $NF}')
    content=$(aws s3 cp "${S3_PATH}${fname}" - --region "$REGION" 2>/dev/null)
    echo "  $fname: $content"
done

if [[ "${1:-}" == "--ssh" ]]; then
    echo
    echo "=== SSH File Counts ==="
    # Get IPs of running instances
    ips=$(aws ec2 describe-instances --region "$REGION" \
        --filters "Name=tag:Project,Values=poker-2026" "Name=instance-state-name,Values=running" \
        --query 'Reservations[*].Instances[*].PublicIpAddress' \
        --output text 2>/dev/null)

    for ip in $ips; do
        files=$(ssh -i ~/.ssh/poker-debug-key.pem -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$ip \
            "ls /opt/blueprint/output_river/*.npz 2>/dev/null | wc -l" 2>/dev/null || echo "unreachable")
        echo "  $ip: $files files"
    done
fi

echo
echo "=== Compute Logs (latest) ==="
for logfile in $(aws s3 ls "${S3_PATH}" --region "$REGION" 2>&1 | grep "log_" | awk '{print $NF}' | head -5); do
    echo "--- $logfile ---"
    aws s3 cp "${S3_PATH}${logfile}" - --region "$REGION" 2>/dev/null | tail -3
done
