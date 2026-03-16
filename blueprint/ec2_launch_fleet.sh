#!/usr/bin/env bash
#
# Launch 8 c5.9xlarge instances for parallel unbucketed blueprint computation.
#
# Each instance receives a user-data script that:
#   1. Installs python3, pip, numpy, numba, awscli
#   2. Downloads code from S3
#   3. Warms up Numba JIT
#   4. Runs compute_unbucketed.py for its assigned cluster range
#   5. Uploads partial results to S3
#   6. Shuts down when complete
#
# Prerequisites:
#   - AWS CLI configured with profile: default
#   - SSH key: ~/.ssh/poker-debug-key.pem
#   - Security group: poker-solver-sg (allows SSH inbound)
#   - Instance profile: default (allows S3 access)
#   - Code tarball uploaded via package_code.sh
#
# Usage:
#   ./ec2_launch_fleet.sh                    # Launch all 8 instances
#   ./ec2_launch_fleet.sh --dry-run          # Print what would be launched
#   ./ec2_launch_fleet.sh --street flop      # Override street (default: flop)
#   ./ec2_launch_fleet.sh --n-instances 4    # Override instance count
#
# Cost estimate (us-east-1, c5.9xlarge on-demand, March 2026):
#   $1.53/hour per instance
#   8 instances x ~6 hours = ~$73 total
#   With 4 instances x ~12 hours = ~$73 total (same cost, longer wall clock)

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

N_INSTANCES=10
TOTAL_BOARDS=2925
STREET="multi_street"
DRY_RUN=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_LIST_FILE="${SCRIPT_DIR}/fleet_instances.txt"

# ============================================================
# Parse arguments
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)       DRY_RUN=true; shift ;;
        --street)        STREET="$2"; shift 2 ;;
        --n-instances)   N_INSTANCES="$2"; shift 2 ;;
        --boards)      TOTAL_BOARDS="$2"; shift 2 ;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--street flop|turn|river] [--n-instances N] [--boards N]"
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

# Compute cluster range for instance $1 of $2 total, splitting $3 boards
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
    # Maybe it's already an ID (sg-xxxx)
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
    range=$(cluster_range "$instance_idx" "$N_INSTANCES" "$TOTAL_BOARDS")
    local board_start board_end
    board_start=$(echo "$range" | cut -d' ' -f1)
    board_end=$(echo "$range" | cut -d' ' -f2)

    cat <<EOF
#!/bin/bash
set -euo pipefail

# Redirect all output to log file
exec > >(tee -a /var/log/blueprint-worker.log) 2>&1

echo "=========================================="
echo "Multi-Street v7 Worker ${instance_idx}"
echo "Boards: ${board_start} to ${board_end}"
echo "Started: \$(date -u)"
echo "=========================================="

export DEBIAN_FRONTEND=noninteractive

# --- Step 1: Install dependencies ---
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv awscli

echo "[2/6] Installing Python packages..."
pip3 install --quiet numpy numba

# --- Step 2: Download code from S3 ---
echo "[3/6] Downloading code from S3..."
mkdir -p /opt/blueprint
cd /opt/blueprint
aws s3 cp ${S3_BUCKET}/code/poker-blueprint.tar.gz /opt/blueprint/code.tar.gz --region ${REGION}
tar xzf code.tar.gz
rm code.tar.gz

# --- Step 3: Warm up Numba JIT ---
echo "[4/6] Warming up Numba JIT..."
cd /opt/blueprint
python3 -c "
import sys
sys.path.insert(0, 'blueprint')
sys.path.insert(0, 'submission')
from multi_street_solver import warmup_jit
warmup_jit()
print('Numba JIT warmup complete')
"

# --- Step 4: Run multi-street computation ---
echo "[5/6] Running compute_multi_street.py (boards ${board_start}-${board_end})..."
echo "COMPUTING" > /opt/blueprint/status.txt
echo "${instance_idx}" > /opt/blueprint/instance_idx.txt

python3 blueprint/compute_multi_street.py \\
    --all_boards \\
    --cluster_start ${board_start} \\
    --cluster_end ${board_end} \\
    --n_iterations 100 \\
    --n_workers 8 \\
    --position_aware \\
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
echo "[6/6] Uploading results to S3..."
aws s3 cp /opt/blueprint/output/ \\
    ${S3_BUCKET}/multi_street_v7/ \\
    --recursive --region ${REGION}

# Upload the log too
aws s3 cp /var/log/blueprint-worker.log \\
    ${S3_BUCKET}/logs/v7_worker_${instance_idx}.log \\
    --region ${REGION}
aws s3 cp /opt/blueprint/compute.log \\
    ${S3_BUCKET}/logs/v7_compute_${instance_idx}.log \\
    --region ${REGION}

echo "UPLOADED" > /opt/blueprint/status.txt

echo "=========================================="
echo "Worker ${instance_idx} complete at \$(date -u)"
echo "=========================================="

# --- Step 6: Self-terminate ---
echo "Shutting down in 60 seconds..."
sleep 60
shutdown -h now
EOF
}

# ============================================================
# Launch instances
# ============================================================

log ""
log "Fleet Launch Configuration"
log "=========================="
log "  Instances:     ${N_INSTANCES} x ${INSTANCE_TYPE}"
log "  AMI:           ${AMI_ID}"
log "  Street:        ${STREET}"
log "  Boards:      ${TOTAL_BOARDS}"
log "  S3 bucket:     ${S3_BUCKET}"
log "  Key pair:      ${KEY_NAME}"
log "  Security grp:  ${SG_ID}"
log "  Profile:       ${INSTANCE_PROFILE}"
log "  Region:        ${REGION}"
log ""

# Show cluster assignments
for i in $(seq 0 $((N_INSTANCES - 1))); do
    range=$(cluster_range "$i" "$N_INSTANCES" "$TOTAL_BOARDS")
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
read -r -p "Launch ${N_INSTANCES} instances? This will cost ~\$$(( N_INSTANCES * 9 ))/hr. [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    log "Aborted."
    exit 0
fi

INSTANCE_IDS=()

for i in $(seq 0 $((N_INSTANCES - 1))); do
    range=$(cluster_range "$i" "$N_INSTANCES" "$TOTAL_BOARDS")
    start=$(echo "$range" | cut -d' ' -f1)
    end=$(echo "$range" | cut -d' ' -f2)

    log "Launching instance $i (boards ${start}-$((end - 1)))..."

    # Write user-data to temp file (avoids shell quoting issues)
    USERDATA_FILE=$(mktemp /tmp/userdata_XXXXXX.sh)
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
                {Key=Name,Value=blueprint-${STREET}-worker-${i}},
                {Key=Project,Value=poker-blueprint-2026},
                {Key=Street,Value=${STREET}},
                {Key=ClusterStart,Value=${start}},
                {Key=ClusterEnd,Value=${end}},
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

# Write instance list to file for use by other scripts
: > "$INSTANCE_LIST_FILE"

for i in $(seq 0 $((N_INSTANCES - 1))); do
    IID="${INSTANCE_IDS[$i]}"
    IP=$(aws ec2 describe-instances \
        --instance-ids "$IID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text \
        --profile "$AWS_PROFILE" \
        --region "$REGION")

    range=$(cluster_range "$i" "$N_INSTANCES" "$TOTAL_BOARDS")
    start=$(echo "$range" | cut -d' ' -f1)
    end=$(echo "$range" | cut -d' ' -f2)

    log "  Worker $i: $IID  $IP  boards ${start}-$((end - 1))"
    echo "${i} ${IID} ${IP} ${start} ${end}" >> "$INSTANCE_LIST_FILE"
done

log ""
log "Instance list saved: ${INSTANCE_LIST_FILE}"
log ""
log "Monitor progress:  ./ec2_monitor.sh"
log "Merge results:     ./ec2_merge_and_deploy.sh"
log ""
log "Manual SSH:"
log "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@<IP>"
log "  tail -f /var/log/blueprint-worker.log"
log ""
log "Terminate all:"
log "  aws ec2 terminate-instances --instance-ids ${INSTANCE_IDS[*]} --profile ${AWS_PROFILE} --region ${REGION}"
