#!/bin/bash

# Parse command line arguments
INCLUDE_LIST=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --incl)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                INCLUDE_LIST+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--incl file1 file2 folder1 ...]"
            exit 1
            ;;
    esac
done

# Check if SOURCE_FOLDER is already set and non-empty
if [ -z "${SOURCE_FOLDER}" ]; then
    read -p "Enter the source folder: " SOURCE_FOLDER
    export SOURCE_FOLDER
fi

# Check if DESTINATION_FOLDER is already set and non-empty
if [ -z "${DESTINATION_FOLDER}" ]; then
    read -p "Enter the destination folder (e.g., user@host:/path): " DESTINATION_FOLDER
    export DESTINATION_FOLDER
fi

# Perform the rsync operation
if [ ${#INCLUDE_LIST[@]} -gt 0 ]; then
    # If include list is provided, sync only those items
    for item in "${INCLUDE_LIST[@]}"; do
        # Check if item exists in source folder
        if [ -e "$SOURCE_FOLDER/$item" ]; then
            echo "Syncing: $item"
            rsync -rlptgoDz --no-perms --no-owner --no-group \
                --exclude '.git' \
                --exclude '__pycache__' \
                --exclude '*.pyc' \
                --exclude 'node_modules' \
                --exclude '.vscode' \
                "$SOURCE_FOLDER/$item" "$DESTINATION_FOLDER/"
        else
            echo "Warning: $SOURCE_FOLDER/$item does not exist, skipping..."
        fi
    done
else
    # If no include list, sync everything (original behavior)
    rsync -rlptgoDz --no-perms --no-owner --no-group \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude 'node_modules' \
        --exclude '.vscode' \
        "$SOURCE_FOLDER" "$DESTINATION_FOLDER"
fi