#!/usr/bin/env bash
#
# Launch EC2 instances for blueprint strategy computation.
#
# This script:
# 1. Launches one or more c5.4xlarge instances (16 vCPUs, 32 GB RAM)
# 2. Installs dependencies and uploads code
# 3. Splits work across instances (each gets a subset of board clusters)
# 4. Monitors progress
# 5. Collects results
# 6. Auto-terminates instances when done (cost control)
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - An SSH key pair registered in your target region
#   - A security group that allows SSH (port 22) inbound
#
# Usage:
#   ./ec2_launch.sh                    # 1 instance, default settings
#   ./ec2_launch.sh --instances 4      # 4 instances, split work
#   ./ec2_launch.sh --dry-run          # print commands without executing
#
# Cost estimate (us-east-1, March 2026):
#   c5.4xlarge: ~$0.68/hour on-demand, ~$0.25/hour spot
#   4 instances x 8 hours = $8-22 total

set -euo pipefail

# ============================================================
# Configuration — edit these for your environment
# ============================================================

# AWS settings
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.4xlarge}"
AMI_ID="${AMI_ID:-ami-0c7217cdde317cfec}"  # Ubuntu 22.04 LTS x86_64 us-east-1
KEY_NAME="${KEY_NAME:-poker-blueprint}"
SECURITY_GROUP="${SECURITY_GROUP:-sg-blueprint}"
SUBNET_ID="${SUBNET_ID:-}"  # Leave empty for default VPC

# Computation settings
N_INSTANCES="${N_INSTANCES:-1}"
N_ITERATIONS="${N_ITERATIONS:-10000}"
N_BUCKETS="${N_BUCKETS:-50}"
N_CLUSTERS="${N_CLUSTERS:-200}"
OUTPUT_BUCKET="${OUTPUT_BUCKET:-s3://poker-blueprint-output}"

# Local paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Runtime settings
USE_SPOT="${USE_SPOT:-true}"
DRY_RUN=false
SKIP_LAUNCH=false

# ============================================================
# Parse arguments
# ============================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --instances)
            N_INSTANCES="$2"
            shift 2
            ;;
        --iterations)
            N_ITERATIONS="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --instance-type)
            INSTANCE_TYPE="$2"
            shift 2
            ;;
        --key-name)
            KEY_NAME="$2"
            shift 2
            ;;
        --security-group)
            SECURITY_GROUP="$2"
            shift 2
            ;;
        --ami)
            AMI_ID="$2"
            shift 2
            ;;
        --output-bucket)
            OUTPUT_BUCKET="$2"
            shift 2
            ;;
        --on-demand)
            USE_SPOT=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-launch)
            SKIP_LAUNCH=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --instances N       Number of EC2 instances (default: 1)"
            echo "  --iterations N      CFR iterations (default: 10000)"
            echo "  --region REGION     AWS region (default: us-east-1)"
            echo "  --instance-type T   Instance type (default: c5.4xlarge)"
            echo "  --key-name NAME     SSH key pair name (default: poker-blueprint)"
            echo "  --security-group SG Security group ID (default: sg-blueprint)"
            echo "  --ami AMI_ID        AMI ID (default: Ubuntu 22.04)"
            echo "  --output-bucket S3  S3 bucket for results (default: s3://poker-blueprint-output)"
            echo "  --on-demand         Use on-demand instead of spot instances"
            echo "  --dry-run           Print commands without executing"
            echo "  --skip-launch       Skip launching, use existing instances"
            echo "  --help              Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================
# Helper functions
# ============================================================

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

run_cmd() {
    if $DRY_RUN; then
        echo "[DRY RUN] $*"
    else
        eval "$@"
    fi
}

# ============================================================
# Generate the user-data bootstrap script
# ============================================================

generate_userdata() {
    local instance_index=$1
    local total_instances=$2

    cat <<USERDATA
#!/bin/bash
set -euo pipefail
exec > /var/log/blueprint-setup.log 2>&1

echo "=== Blueprint computation setup starting ==="
echo "Instance ${instance_index} of ${total_instances}"

# System setup
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv awscli

# Create work directory
mkdir -p /opt/blueprint
cd /opt/blueprint

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --quiet numpy

# Signal that setup is complete
echo "SETUP_COMPLETE" > /opt/blueprint/status.txt

echo "=== Setup complete, waiting for code upload ==="
USERDATA
}

# ============================================================
# Generate the computation script that runs on each instance
# ============================================================

generate_compute_script() {
    local instance_index=$1
    local total_instances=$2

    cat <<'COMPUTE'
#!/bin/bash
set -euo pipefail
exec > /opt/blueprint/compute.log 2>&1

cd /opt/blueprint
source venv/bin/activate

echo "=== Starting blueprint computation ==="
echo "Instance: INSTANCE_INDEX of TOTAL_INSTANCES"
echo "Iterations: N_ITERATIONS_VAL"
echo "Buckets: N_BUCKETS_VAL"
echo "Clusters: N_CLUSTERS_VAL"

# Calculate this instance's share of boards
# Total representative boards are determined by n_clusters
# Each instance gets clusters: [start, end)
TOTAL_CLUSTERS=N_CLUSTERS_VAL
CLUSTERS_PER_INSTANCE=$(( (TOTAL_CLUSTERS + TOTAL_INSTANCES - 1) / TOTAL_INSTANCES ))
START=$(( INSTANCE_INDEX * CLUSTERS_PER_INSTANCE ))
END=$(( START + CLUSTERS_PER_INSTANCE ))
if [ $END -gt $TOTAL_CLUSTERS ]; then
    END=$TOTAL_CLUSTERS
fi

N_BOARDS=$(( END - START ))
if [ $N_BOARDS -le 0 ]; then
    echo "No boards to process for this instance."
    echo "COMPUTATION_COMPLETE" > /opt/blueprint/status.txt
    exit 0
fi

echo "Processing boards $START to $END ($N_BOARDS boards)"
echo "COMPUTING" > /opt/blueprint/status.txt

# Run the computation
# Each instance uses all its CPUs (c5.4xlarge = 16 vCPUs)
python3 /opt/blueprint/code/blueprint/compute_blueprint.py \
    --n_iterations N_ITERATIONS_VAL \
    --n_buckets N_BUCKETS_VAL \
    --n_clusters N_CLUSTERS_VAL \
    --n_boards $N_BOARDS \
    --n_workers 16 \
    --output_dir /opt/blueprint/output

echo "=== Computation complete ==="
echo "COMPUTATION_COMPLETE" > /opt/blueprint/status.txt

# Upload results to S3
if command -v aws &> /dev/null; then
    aws s3 cp /opt/blueprint/output/ OUTPUT_BUCKET_VAL/instance_INSTANCE_INDEX/ --recursive
    echo "UPLOADED" > /opt/blueprint/status.txt
    echo "Results uploaded to OUTPUT_BUCKET_VAL/instance_INSTANCE_INDEX/"
fi

# Auto-terminate after completion (cost control)
echo "Scheduling self-termination in 5 minutes..."
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
(sleep 300 && aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region REGION_VAL) &
COMPUTE

    # Replace placeholders
    sed -e "s/INSTANCE_INDEX/${instance_index}/g" \
        -e "s/TOTAL_INSTANCES/${total_instances}/g" \
        -e "s/N_ITERATIONS_VAL/${N_ITERATIONS}/g" \
        -e "s/N_BUCKETS_VAL/${N_BUCKETS}/g" \
        -e "s/N_CLUSTERS_VAL/${N_CLUSTERS}/g" \
        -e "s|OUTPUT_BUCKET_VAL|${OUTPUT_BUCKET}|g" \
        -e "s/REGION_VAL/${REGION}/g"
}

# ============================================================
# Main execution
# ============================================================

log "Blueprint EC2 Launcher"
log "======================"
log "Region:        $REGION"
log "Instance type: $INSTANCE_TYPE"
log "Instances:     $N_INSTANCES"
log "Iterations:    $N_ITERATIONS"
log "Buckets:       $N_BUCKETS"
log "Clusters:      $N_CLUSTERS"
log "Spot:          $USE_SPOT"
log "Output:        $OUTPUT_BUCKET"
log ""

# Check prerequisites
if ! command -v aws &> /dev/null; then
    echo "ERROR: AWS CLI not found. Install with: pip install awscli"
    exit 1
fi

# Verify AWS credentials
if ! aws sts get-caller-identity --region "$REGION" > /dev/null 2>&1; then
    echo "ERROR: AWS credentials not configured. Run: aws configure"
    exit 1
fi

# Create S3 bucket if it doesn't exist
log "Ensuring S3 bucket exists..."
BUCKET_NAME=$(echo "$OUTPUT_BUCKET" | sed 's|s3://||')
run_cmd "aws s3 mb '$OUTPUT_BUCKET' --region '$REGION' 2>/dev/null || true"

# Package the code for upload
log "Packaging code..."
TARBALL="/tmp/blueprint-code.tar.gz"
run_cmd "tar czf '$TARBALL' -C '$PROJECT_DIR' \
    blueprint/ \
    submission/equity.py \
    submission/game_tree.py \
    submission/solver.py \
    submission/inference.py \
    submission/data/"

TARBALL_SIZE=$(du -h "$TARBALL" 2>/dev/null | cut -f1 || echo "unknown")
log "Code package: $TARBALL ($TARBALL_SIZE)"

# Upload code to S3 for instances to fetch
run_cmd "aws s3 cp '$TARBALL' '$OUTPUT_BUCKET/code/blueprint-code.tar.gz' --region '$REGION'"

# Launch instances
INSTANCE_IDS=()

for i in $(seq 0 $((N_INSTANCES - 1))); do
    log "Launching instance $((i+1))/$N_INSTANCES..."

    USERDATA=$(generate_userdata $i $N_INSTANCES | base64)

    LAUNCH_ARGS=(
        "--image-id" "$AMI_ID"
        "--instance-type" "$INSTANCE_TYPE"
        "--key-name" "$KEY_NAME"
        "--security-group-ids" "$SECURITY_GROUP"
        "--user-data" "$USERDATA"
        "--region" "$REGION"
        "--tag-specifications" "ResourceType=instance,Tags=[{Key=Name,Value=blueprint-worker-$i},{Key=Project,Value=poker-blueprint}]"
        "--iam-instance-profile" "Name=blueprint-worker-role"
    )

    if [ -n "$SUBNET_ID" ]; then
        LAUNCH_ARGS+=("--subnet-id" "$SUBNET_ID")
    fi

    if $USE_SPOT; then
        LAUNCH_ARGS+=("--instance-market-options" "MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}")
    fi

    if $DRY_RUN; then
        log "[DRY RUN] aws ec2 run-instances ${LAUNCH_ARGS[*]}"
        INSTANCE_IDS+=("i-dry-run-$i")
    else
        INSTANCE_ID=$(aws ec2 run-instances "${LAUNCH_ARGS[@]}" \
            --query 'Instances[0].InstanceId' --output text)
        INSTANCE_IDS+=("$INSTANCE_ID")
        log "  Instance launched: $INSTANCE_ID"
    fi
done

log ""
log "All ${N_INSTANCES} instances launched."
log "Instance IDs: ${INSTANCE_IDS[*]}"

# Wait for instances to be running
log "Waiting for instances to enter running state..."
for iid in "${INSTANCE_IDS[@]}"; do
    if ! $DRY_RUN; then
        aws ec2 wait instance-running --instance-ids "$iid" --region "$REGION"
    fi
done
log "All instances running."

# Get public IPs
declare -A INSTANCE_IPS
for idx in $(seq 0 $((N_INSTANCES - 1))); do
    iid="${INSTANCE_IDS[$idx]}"
    if ! $DRY_RUN; then
        IP=$(aws ec2 describe-instances --instance-ids "$iid" --region "$REGION" \
            --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
        INSTANCE_IPS[$idx]="$IP"
        log "  Instance $idx: $iid -> $IP"
    fi
done

# Wait for SSH to be available and upload code
log ""
log "Uploading code and starting computation..."

for idx in $(seq 0 $((N_INSTANCES - 1))); do
    if $DRY_RUN; then
        log "[DRY RUN] Would upload code and start computation on instance $idx"
        continue
    fi

    IP="${INSTANCE_IPS[$idx]}"
    SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ~/.ssh/${KEY_NAME}.pem"

    # Wait for SSH
    log "  Waiting for SSH on $IP..."
    for attempt in $(seq 1 30); do
        if ssh $SSH_OPTS ubuntu@"$IP" "echo ok" 2>/dev/null; then
            break
        fi
        sleep 10
    done

    # Wait for userdata setup to complete
    log "  Waiting for setup to complete on $IP..."
    for attempt in $(seq 1 60); do
        STATUS=$(ssh $SSH_OPTS ubuntu@"$IP" "cat /opt/blueprint/status.txt 2>/dev/null || echo PENDING")
        if [ "$STATUS" = "SETUP_COMPLETE" ]; then
            break
        fi
        sleep 10
    done

    # Upload code
    log "  Uploading code to $IP..."
    scp $SSH_OPTS "$TARBALL" ubuntu@"$IP":/tmp/blueprint-code.tar.gz
    ssh $SSH_OPTS ubuntu@"$IP" "
        mkdir -p /opt/blueprint/code
        tar xzf /tmp/blueprint-code.tar.gz -C /opt/blueprint/code/
    "

    # Generate and upload compute script
    COMPUTE_SCRIPT=$(generate_compute_script $idx $N_INSTANCES)
    echo "$COMPUTE_SCRIPT" | ssh $SSH_OPTS ubuntu@"$IP" "cat > /opt/blueprint/run_compute.sh && chmod +x /opt/blueprint/run_compute.sh"

    # Start computation in background
    log "  Starting computation on $IP (instance $idx)..."
    ssh $SSH_OPTS ubuntu@"$IP" "nohup /opt/blueprint/run_compute.sh > /opt/blueprint/compute.log 2>&1 &"
done

log ""
log "========================================"
log "All instances started!"
log "========================================"
log ""
log "Monitor progress:"
log "  For each instance, SSH in and check:"
log "    ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@<IP>"
log "    tail -f /opt/blueprint/compute.log"
log "    cat /opt/blueprint/status.txt"
log ""
log "Results will be uploaded to: $OUTPUT_BUCKET"
log ""
log "To collect results after completion:"
log "  aws s3 sync $OUTPUT_BUCKET/instance_0/ ./blueprint_results/ --region $REGION"
if [ "$N_INSTANCES" -gt 1 ]; then
    log "  # For multiple instances, merge results from all:"
    for idx in $(seq 0 $((N_INSTANCES - 1))); do
        log "  aws s3 sync $OUTPUT_BUCKET/instance_${idx}/ ./blueprint_results/instance_${idx}/ --region $REGION"
    done
fi
log ""
log "COST CONTROL: Instances will auto-terminate 5 minutes after computation completes."
log "To manually terminate all instances:"
log "  aws ec2 terminate-instances --instance-ids ${INSTANCE_IDS[*]} --region $REGION"
log ""

# Save instance info for later reference
INFO_FILE="/tmp/blueprint-instances.txt"
{
    echo "Blueprint Computation - $(date)"
    echo "Region: $REGION"
    echo "Instance Type: $INSTANCE_TYPE"
    echo "Instances: ${N_INSTANCES}"
    echo ""
    for idx in $(seq 0 $((N_INSTANCES - 1))); do
        echo "Instance $idx: ${INSTANCE_IDS[$idx]} ${INSTANCE_IPS[$idx]:-N/A}"
    done
    echo ""
    echo "Output: $OUTPUT_BUCKET"
} > "$INFO_FILE"
log "Instance info saved to: $INFO_FILE"
