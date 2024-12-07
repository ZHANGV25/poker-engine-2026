#!/bin/bash

# Get current date in YYYYMMDD format
current_date=$(date +%Y%m%d)

# Create zip file name
zip_name="aipoker-${current_date}.zip"

# Create zip file while excluding unwanted files
zip -r "$zip_name" \
    gym_env.py \
    run.py \
    requirements.txt \
    agents/ \
    -x "**/.DS_Store" \
    "**/__pycache__/*" \
    "**/*.pyc" \
    "**/*.pyo" \
    "**/*.pyd" \
    "**/.git/*" \
    "**/.idea/*" \
    "**/.vscode/*"

echo "Created $zip_name successfully!"

aws s3 cp "$zip_name" "s3://cmu-poker-releases/$zip_name"

if [ $? -eq 0 ]; then
    echo "Successfully uploaded $zip_name to S3 bucket cmu-poker-releases"
    rm "$zip_name"
else
    echo "Failed to upload $zip_name to S3"
    exit 1
fi