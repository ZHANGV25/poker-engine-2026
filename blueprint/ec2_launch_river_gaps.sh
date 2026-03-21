#!/usr/bin/env bash
#
# Launch EC2 fleet to fill gaps in river precompute.
#
# 15 instances with 32 workers each (vs original 8), covering:
#   Gap 1: [0, 16146)       — 4 instances
#   Gap 2: [24219, 28255)   — 1 instance
#   Gap 3: [32292, 48438)   — 4 instances
#   Gap 4: [56511, 80730)   — 6 instances
#
# Estimated time: ~2.5 hours per instance at 32 workers.
#
# Usage:
#   bash blueprint/package_code.sh                # Upload code first
#   bash blueprint/ec2_launch_river_gaps.sh       # Launch fleet

set -euo pipefail

AWS_PROFILE="default"
REGION="us-east-1"
INSTANCE_TYPE="c5.9xlarge"
AMI_ID="ami-0c7217cdde317cfec"
KEY_NAME="poker-debug-key"
SECURITY_GROUP="poker-solver-sg"
INSTANCE_PROFILE="poker-solver-profile"
S3_BUCKET="s3://poker-blueprint-2026"

ITERS=500
N_WORKERS=32

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_LIST_FILE="${SCRIPT_DIR}/fleet_river_gaps_instances.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Define the 15 gap ranges
declare -a RANGES=(
    "0 4037"
    "4037 8073"
    "8073 12110"
    "12110 16146"
    "24219 28255"
    "32292 36329"
    "36329 40365"
    "40365 44402"
    "44402 48438"
    "56511 60548"
    "60548 64584"
    "64584 68621"
    "68621 72657"
    "72657 76694"
    "76694 80730"
)

N_INSTANCES=${#RANGES[@]}

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
    local board_start=$2
    local board_end=$3

    cat <<EOF
#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/river-worker.log) 2>&1

echo "=========================================="
echo "River Gap Worker ${instance_idx}"
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

# Save instance index
echo "${instance_idx}" > /opt/blueprint/instance_idx.txt

# Set up periodic S3 sync every 15 minutes
(crontab -l 2>/dev/null; echo "*/15 * * * * aws s3 sync /opt/blueprint/output_river/ ${S3_BUCKET}/river_v2/ --region ${REGION} --quiet >> /tmp/s3-sync.log 2>&1") | crontab -
(crontab -l 2>/dev/null; echo "*/15 * * * * echo \"\\\$(date): \\\$(ls /opt/blueprint/output_river/*.npz 2>/dev/null | wc -l) files done\" > /tmp/status.txt; aws s3 cp /tmp/status.txt ${S3_BUCKET}/river_v2/status_gap_${instance_idx}.txt --region ${REGION} --quiet 2>/dev/null") | crontab -

# Run river precompute
echo "COMPUTING" > /opt/blueprint/status.txt

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

# Final upload
aws s3 sync /opt/blueprint/output_river/ \\
    ${S3_BUCKET}/river_v2/ \\
    --region ${REGION} \\
    --quiet

echo "UPLOAD_DONE" > /opt/blueprint/status.txt
aws s3 cp /opt/blueprint/status.txt ${S3_BUCKET}/river_v2/status_gap_${instance_idx}.txt --region ${REGION}
aws s3 cp /opt/blueprint/compute.log ${S3_BUCKET}/river_v2/log_gap_${instance_idx}.txt --region ${REGION}

echo "All done. Shutting down."
shutdown -h now
EOF
}

# Launch instances
log "Launching ${N_INSTANCES} x ${INSTANCE_TYPE} instances (${N_WORKERS} workers each)..."
> "$INSTANCE_LIST_FILE"

for i in $(seq 0 $((N_INSTANCES - 1))); do
    range="${RANGES[$i]}"
    board_start=$(echo "$range" | cut -d' ' -f1)
    board_end=$(echo "$range" | cut -d' ' -f2)

    USERDATA=$(generate_userdata "$i" "$board_start" "$board_end")

    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --iam-instance-profile Name="$INSTANCE_PROFILE" \
        --user-data "$USERDATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=river-gap-${i}},{Key=Project,Value=poker-2026}]" \
        --query 'Instances[0].InstanceId' \
        --output text \
        --profile "$AWS_PROFILE" \
        --region "$REGION" 2>/dev/null)

    if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
        log "  FAILED to launch instance ${i} (boards ${board_start}-${board_end})"
        echo "${i} FAILED FAILED ${board_start} ${board_end}" >> "$INSTANCE_LIST_FILE"
        continue
    fi

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
log "Estimated completion: ~2.5 hours"
log "Check progress: aws s3 ls s3://poker-blueprint-2026/river_v2/ --region us-east-1 | wc -l"
