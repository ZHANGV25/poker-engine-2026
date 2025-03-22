#!/bin/bash

# Get current date in YYYYMMDD format
current_date=$(date +%Y%m%d)

# Create zip file name
zip_name="aipoker-${current_date}.zip"

# Create temporary folder
temp_dir="aipoker"
rm -rf "$temp_dir"
mkdir -p "$temp_dir/agents" "$temp_dir/submission" "$temp_dir/docs"

# Copy files and folders to temporary directory, maintaining structure
cp gym_env.py match.py run.py agent_test.py requirements.txt "$temp_dir/"
cp agents/agent.py "$temp_dir/agents/"
cp agents/test_agents.py "$temp_dir/agents/"
cp -r submission/* "$temp_dir/submission/"
cp -r docs/* "$temp_dir/docs/"
cp agent_config.json "$temp_dir/"

# Create zip file while excluding unwanted files
zip -r "$zip_name" "$temp_dir" \
    -x "**/.DS_Store" \
    "**/__pycache__/*" \
    "**/*.pyc" \
    "**/*.pyo" \
    "**/*.pyd" \
    "**/.git/*" \
    "**/.idea/*" \
    "**/.vscode/*"

echo "Created $zip_name successfully!"

# Upload to S3
aws s3 cp "$zip_name" "s3://cmu-poker-releases/$zip_name"

if [ $? -eq 0 ]; then
    echo "Successfully uploaded $zip_name to S3 bucket cmu-poker-releases"
    # Clean up local files
    rm "$zip_name"
    rm -rf "$temp_dir"
    echo "Cleaned up local files"
else
    echo "Failed to upload $zip_name to S3"
    rm -rf "$temp_dir"
    exit 1
fi