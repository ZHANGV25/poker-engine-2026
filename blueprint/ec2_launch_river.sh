#!/usr/bin/env bash
#
# Launch 10 c5.9xlarge instances for parallel river strategy computation.
#
# Distributes 80,730 river boards (C(27,5)) across 10 instances evenly.
# Each instance runs compute_river.py for its assigned board range,
# then uploads results to S3.
#
# Prerequisites:
#   - AWS CLI configured with profile: default
#   - SSH key: ~/.ssh/poker-debug-key.pem
#   - Security group: poker-solver-sg (allows SSH inbound)
#   - Instance profile: poker-solver-profile (allows S3 access)
#   - Code tarball uploaded via package_code.sh
#
# Usage:
#   ./ec2_launch_river.sh                  # Launch all 10 instances
#   ./ec2_launch_river.sh --dry-run        # Print what would be launched
#   ./ec2_launch_river.sh --n-instances 5  # Override instance count
#
# Cost estimate (us-east-1, c5.9xlarge on-demand, March 2026):
#   $1.53/hour per instance
#   10 instances x ~4 hours = ~$61 total

set -euo pipefail

# ============================================================
# Configuration
# ============================================================

AWS_PROFILE="default"
REGION="us-east-1"
INSTANCE_TYPE="c5.9xlarge"
AMI_ID="ami-0c7217cdde317cfec"  # Ubuntu 22.04 LTS x86_64 us-east-1
KEY_NAME="poker-debug-key"
SECURITY_GROUP="poker-solver-sg"
INSTANCE_PROFILE="poker-solver-profile"
S3_BUCKET="s3://poker-blueprint-2026"
S3_RESULT_PREFIX="river_v1"

N_INSTANCES=10
TOTAL_BOARDS=80730
N_WORKERS=34       # 36 vCPU - 2 reserved for OS
CFR_ITERATIONS=2000
DRY_RUN=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_LIST_FILE="${SCRIPT_DIR}/fleet_river_instances.txt"

# ============================================================
# Parse arguments
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)       DRY_RUN=true; shift ;;
        --n-instances)   N_INSTANCES="$2"; shift 2 ;;
        --iterations)    CFR_ITERATIONS="$2"; shift 2 ;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--n-instances N] [--iterations N] [--instance-type TYPE]"
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

# Compute board range for instance $1 of $2 total, splitting $3 boards
board_range() {
    local idx=$1 total=$2 n_boards=$3
    local per_instance=$(( (n_boards + total - 1) / total ))
    local start=$(( idx * per_instance ))
    local end=$(( start + per_instance ))
    if (( end > n_boards )); then
        end=$n_boards
    fi
    echo "${start} ${end}"
}

# ============================================================
# Preflight checks
# ============================================================

log "Preflight checks..."

if ! command -v aws &>/dev/null; then
    echo "ERROR: AWS CLI not installed."
    exit 1
fi

if ! aws sts get-caller-identity --profile "$AWS_PROFILE" --region "$REGION" &>/dev/null; then
    echo "ERROR: AWS credentials invalid for profile '$AWS_PROFILE'."
    exit 1
fi

# Verify code tarball exists in S3
if ! aws s3 ls "${S3_BUCKET}/code/poker-blueprint.tar.gz" \
    --profile "$AWS_PROFILE" --region "$REGION" &>/dev/null; then
    echo "ERROR: Code tarball not found at ${S3_BUCKET}/code/poker-blueprint.tar.gz"
    echo "Run package_code.sh first."
    exit 1
fi

# Resolve security group name to ID if needed
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

log "Security group resolved: $SECURITY_GROUP -> $SG_ID"

# ============================================================
# Generate user-data script for each instance
# ============================================================

generate_userdata() {
    local instance_idx=$1
    local range
    range=$(board_range "$instance_idx" "$N_INSTANCES" "$TOTAL_BOARDS")
    local board_start board_end
    board_start=$(echo "$range" | cut -d' ' -f1)
    board_end=$(echo "$range" | cut -d' ' -f2)

    cat <<EOF
#!/bin/bash
set -euo pipefail

# Redirect all output to log file
exec > >(tee -a /var/log/river-worker.log) 2>&1

echo "=========================================="
echo "River Strategy Worker ${instance_idx}"
echo "Boards: ${board_start} to ${board_end}"
echo "Started: \$(date -u)"
echo "=========================================="

export DEBIAN_FRONTEND=noninteractive

# --- Step 1: Install dependencies ---
echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv awscli

echo "[2/5] Installing Python packages..."
pip3 install --quiet numpy numba

# --- Step 2: Download code from S3 ---
echo "[3/5] Downloading code from S3..."
mkdir -p /opt/blueprint
cd /opt/blueprint
aws s3 cp ${S3_BUCKET}/code/poker-blueprint.tar.gz /opt/blueprint/code.tar.gz --region ${REGION}
tar xzf code.tar.gz
rm code.tar.gz

# --- Step 3: Warm up Numba JIT ---
echo "[4/5] Warming up Numba JIT..."
cd /opt/blueprint
python3 -c "
import sys
sys.path.insert(0, 'blueprint')
sys.path.insert(0, 'submission')
from multi_street_solver import warmup_jit
warmup_jit()
print('Numba JIT warmup complete')
"

# --- Step 4: Run river computation ---
echo "[5/5] Running compute_river.py (boards ${board_start}-${board_end})..."
echo "COMPUTING" > /opt/blueprint/status.txt
echo "${instance_idx}" > /opt/blueprint/instance_idx.txt

mkdir -p /opt/blueprint/output

python3 blueprint/compute_river.py \\
    --start_board ${board_start} \\
    --end_board ${board_end} \\
    --workers ${N_WORKERS} \\
    --iterations ${CFR_ITERATIONS} \\
    --output_dir /opt/blueprint/output \\
    2>&1 | tee /opt/blueprint/compute.log

COMPUTE_EXIT=\$?

if [ \$COMPUTE_EXIT -eq 0 ]; then
    echo "COMPUTE_DONE" > /opt/blueprint/status.txt
else
    echo "COMPUTE_FAILED (exit \$COMPUTE_EXIT)" > /opt/blueprint/status.txt
    echo "Computation failed with exit code \$COMPUTE_EXIT"
fi

# --- Step 5: Upload results to S3 ---
echo "Uploading results to S3..."
aws s3 cp /opt/blueprint/output/ \\
    ${S3_BUCKET}/${S3_RESULT_PREFIX}/ \\
    --recursive --region ${REGION}

# Upload logs
aws s3 cp /var/log/river-worker.log \\
    ${S3_BUCKET}/logs/river_worker_${instance_idx}.log \\
    --region ${REGION}
aws s3 cp /opt/blueprint/compute.log \\
    ${S3_BUCKET}/logs/river_compute_${instance_idx}.log \\
    --region ${REGION}

echo "UPLOADED" > /opt/blueprint/status.txt

echo "=========================================="
echo "Worker ${instance_idx} complete at \$(date -u)"
echo "=========================================="

# --- Self-terminate ---
echo "Shutting down in 60 seconds..."
sleep 60
shutdown -h now
EOF
}

# ============================================================
# Launch instances
# ============================================================

log ""
log "River Fleet Launch Configuration"
log "================================"
log "  Instances:     ${N_INSTANCES} x ${INSTANCE_TYPE}"
log "  AMI:           ${AMI_ID}"
log "  Total boards:  ${TOTAL_BOARDS}"
log "  CFR iters:     ${CFR_ITERATIONS}"
log "  Workers/inst:  ${N_WORKERS}"
log "  S3 bucket:     ${S3_BUCKET}"
log "  S3 results:    ${S3_BUCKET}/${S3_RESULT_PREFIX}/"
log "  Key pair:      ${KEY_NAME}"
log "  Security grp:  ${SG_ID}"
log "  Profile:       ${INSTANCE_PROFILE}"
log "  Region:        ${REGION}"
log ""

# Show board assignments
for i in $(seq 0 $((N_INSTANCES - 1))); do
    range=$(board_range "$i" "$N_INSTANCES" "$TOTAL_BOARDS")
    start=$(echo "$range" | cut -d' ' -f1)
    end=$(echo "$range" | cut -d' ' -f2)
    count=$((end - start))
    log "  Instance $i: boards ${start}-$((end - 1)) (${count} boards)"
done
log ""

if $DRY_RUN; then
    log "[DRY RUN] Would launch ${N_INSTANCES} instances. Exiting."
    exit 0
fi

# Confirm before launching
HOURLY_COST=$(echo "$N_INSTANCES * 1.53" | bc 2>/dev/null || echo "$((N_INSTANCES * 2))")
read -r -p "Launch ${N_INSTANCES} instances? Estimated ~\$${HOURLY_COST}/hr. [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    log "Aborted."
    exit 0
fi

INSTANCE_IDS=()

for i in $(seq 0 $((N_INSTANCES - 1))); do
    range=$(board_range "$i" "$N_INSTANCES" "$TOTAL_BOARDS")
    start=$(echo "$range" | cut -d' ' -f1)
    end=$(echo "$range" | cut -d' ' -f2)

    log "Launching instance $i (boards ${start}-$((end - 1)))..."

    # Write user-data to temp file (avoids shell quoting issues)
    USERDATA_FILE=$(mktemp /tmp/userdata_river_XXXXXX.sh)
    generate_userdata "$i" > "$USERDATA_FILE"

    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --iam-instance-profile "Name=${INSTANCE_PROFILE}" \
        --user-data "file://${USERDATA_FILE}" \
        --tag-specifications \
            "ResourceType=instance,Tags=[
                {Key=Name,Value=blueprint-river-worker-${i}},
                {Key=Project,Value=poker-blueprint-2026},
                {Key=Street,Value=river},
                {Key=BoardStart,Value=${start}},
                {Key=BoardEnd,Value=${end}},
                {Key=WorkerIndex,Value=${i}}
            ]" \
        --profile "$AWS_PROFILE" \
        --region "$REGION" \
        --query 'Instances[0].InstanceId' \
        --output text)

    INSTANCE_IDS+=("$INSTANCE_ID")
    rm -f "$USERDATA_FILE"

    log "  Launched: $INSTANCE_ID"
done

# Wait for all instances to be running
log ""
log "Waiting for instances to enter 'running' state..."
aws ec2 wait instance-running \
    --instance-ids "${INSTANCE_IDS[@]}" \
    --profile "$AWS_PROFILE" \
    --region "$REGION"
log "All instances running."

# Retrieve public IPs
log ""
log "=========================================="
log "Fleet Summary"
log "=========================================="

# Write instance list to file
: > "$INSTANCE_LIST_FILE"

for i in $(seq 0 $((N_INSTANCES - 1))); do
    IID="${INSTANCE_IDS[$i]}"
    IP=$(aws ec2 describe-instances \
        --instance-ids "$IID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text \
        --profile "$AWS_PROFILE" \
        --region "$REGION")

    range=$(board_range "$i" "$N_INSTANCES" "$TOTAL_BOARDS")
    start=$(echo "$range" | cut -d' ' -f1)
    end=$(echo "$range" | cut -d' ' -f2)

    log "  Worker $i: $IID  $IP  boards ${start}-$((end - 1))"
    echo "${i} ${IID} ${IP} ${start} ${end}" >> "$INSTANCE_LIST_FILE"
done

log ""
log "Instance list saved: ${INSTANCE_LIST_FILE}"
log ""
log "Monitor progress:"
log "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@<IP> tail -f /var/log/river-worker.log"
log ""
log "Results will upload to: ${S3_BUCKET}/${S3_RESULT_PREFIX}/"
log ""
log "Terminate all:"
log "  aws ec2 terminate-instances --instance-ids ${INSTANCE_IDS[*]} --profile ${AWS_PROFILE} --region ${REGION}"
