#!/usr/bin/env bash
#
# Launch EC2 fleet to precompute river strategies for all 80,730 boards.
# 10 x c5.9xlarge instances, 8 workers each, ~500 DCFR iterations.
# Estimated time: 6-8 hours.
#
# Prerequisites:
#   - Run package_code.sh first to upload code to S3
#   - AWS CLI configured, SSH key at ~/.ssh/poker-debug-key.pem
#
# Usage:
#   bash blueprint/package_code.sh                  # Upload code first
#   bash blueprint/ec2_launch_river_fleet.sh        # Launch fleet
#   bash blueprint/ec2_monitor.sh                   # Monitor progress

set -euo pipefail

AWS_PROFILE="default"
REGION="us-east-1"
INSTANCE_TYPE="c5.9xlarge"
AMI_ID="ami-0c7217cdde317cfec"
KEY_NAME="poker-debug-key"
SECURITY_GROUP="poker-solver-sg"
INSTANCE_PROFILE="poker-solver-profile"
S3_BUCKET="s3://poker-blueprint-2026"

N_INSTANCES=10
TOTAL_BOARDS=80730
ITERS=500
N_WORKERS=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_LIST_FILE="${SCRIPT_DIR}/fleet_river_v2_instances.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cluster_range() {
    local idx=$1 total=$2 n_boards=$3
    local per_instance=$(( (n_boards + total - 1) / total ))
    local start=$(( idx * per_instance ))
    local end=$(( start + per_instance ))
    if (( end > n_boards )); then
        end=$n_boards
    fi
    echo "${start} ${end}"
}

# Preflight
log "Preflight checks..."

if ! aws sts get-caller-identity --profile "$AWS_PROFILE" --region "$REGION" &>/dev/null; then
    echo "ERROR: AWS credentials invalid."
    exit 1
fi

# Package and upload code
log "Packaging code..."
bash "${SCRIPT_DIR}/package_code.sh"

SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=${SECURITY_GROUP}" \
    --query 'SecurityGroups[0].GroupId' --output text \
    --profile "$AWS_PROFILE" --region "$REGION" 2>/dev/null || echo "")

if [[ -z "$SG_ID" || "$SG_ID" == "None" ]]; then
    if [[ "$SECURITY_GROUP" == sg-* ]]; then
        SG_ID="$SECURITY_GROUP"
    else
        echo "ERROR: Security group '$SECURITY_GROUP' not found."
        exit 1
    fi
fi

generate_userdata() {
    local instance_idx=$1
    local range
    range=$(cluster_range "$instance_idx" "$N_INSTANCES" "$TOTAL_BOARDS")
    local board_start board_end
    board_start=$(echo "$range" | cut -d' ' -f1)
    board_end=$(echo "$range" | cut -d' ' -f2)

    cat <<EOF
#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/river-worker.log) 2>&1

echo "=========================================="
echo "River Precompute Worker ${instance_idx}"
echo "Boards: ${board_start} to ${board_end}"
echo "Iterations: ${ITERS}"
echo "Workers: ${N_WORKERS}"
echo "Started: \$(date -u)"
echo "=========================================="

export DEBIAN_FRONTEND=noninteractive

# Install
apt-get update -qq
apt-get install -y -qq python3 python3-pip awscli
pip3 install --quiet numpy

# Download code
mkdir -p /opt/blueprint
cd /opt/blueprint
aws s3 cp ${S3_BUCKET}/code/poker-blueprint.tar.gz /opt/blueprint/code.tar.gz --region ${REGION}
tar xzf code.tar.gz
rm code.tar.gz

# Run river precompute
echo "COMPUTING" > /opt/blueprint/status.txt
echo "${instance_idx}" > /opt/blueprint/instance_idx.txt

python3 blueprint/compute_river_strategies.py \\
    --start ${board_start} \\
    --end ${board_end} \\
    --n_workers ${N_WORKERS} \\
    --iters ${ITERS} \\
    --output_dir /opt/blueprint/output_river \\
    2>&1 | tee /opt/blueprint/compute.log

COMPUTE_EXIT=\$?

if [ \$COMPUTE_EXIT -eq 0 ]; then
    echo "COMPUTE_DONE" > /opt/blueprint/status.txt
else
    echo "COMPUTE_FAILED (exit \$COMPUTE_EXIT)" > /opt/blueprint/status.txt
fi

# Upload results
aws s3 sync /opt/blueprint/output_river/ \\
    ${S3_BUCKET}/river_v2/ \\
    --region ${REGION} \\
    --quiet

echo "UPLOAD_DONE" > /opt/blueprint/status.txt
aws s3 cp /opt/blueprint/status.txt ${S3_BUCKET}/river_v2/status_${instance_idx}.txt --region ${REGION}
aws s3 cp /opt/blueprint/compute.log ${S3_BUCKET}/river_v2/log_${instance_idx}.txt --region ${REGION}

echo "All done. Shutting down."
shutdown -h now
EOF
}

# Launch instances
log "Launching ${N_INSTANCES} x ${INSTANCE_TYPE} instances..."
> "$INSTANCE_LIST_FILE"

for i in $(seq 0 $((N_INSTANCES - 1))); do
    range=$(cluster_range "$i" "$N_INSTANCES" "$TOTAL_BOARDS")
    board_start=$(echo "$range" | cut -d' ' -f1)
    board_end=$(echo "$range" | cut -d' ' -f2)

    USERDATA=$(generate_userdata "$i")

    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --iam-instance-profile Name="$INSTANCE_PROFILE" \
        --user-data "$USERDATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=river-worker-${i}},{Key=Project,Value=poker-2026}]" \
        --query 'Instances[0].InstanceId' \
        --output text \
        --profile "$AWS_PROFILE" \
        --region "$REGION" 2>/dev/null)

    # Get public IP (may take a moment)
    sleep 2
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text \
        --profile "$AWS_PROFILE" \
        --region "$REGION" 2>/dev/null || echo "pending")

    echo "${i} ${INSTANCE_ID} ${PUBLIC_IP} ${board_start} ${board_end}" >> "$INSTANCE_LIST_FILE"
    log "  Instance ${i}: ${INSTANCE_ID} (${PUBLIC_IP}) boards ${board_start}-${board_end}"
done

log "Fleet launched! Instance list: ${INSTANCE_LIST_FILE}"
log "Monitor with: bash blueprint/ec2_monitor.sh"
log "Estimated completion: ~6-8 hours"
log "Cost: ~\$92 (10 x \$1.53/hr x 6hr)"
