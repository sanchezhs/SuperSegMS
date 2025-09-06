#!/bin/bash
set -e

ZIP_FILE="project.zip"
OUTPUT_FILE="../paper/output.pdf"
PAPER_DIR="../paper"
MAIN_FILE="main.tex"
ENGINE="xelatex"
DEBUG_URL="http://localhost:8000/debug"
COMPILE_URL="http://localhost:8000/compile"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log "Starting paper compilation process."

# Clean up old files
log "Removing old zip file if it exists."
rm -f "$ZIP_FILE"

# Create a new zip file
log "Creating zip file from $PAPER_DIR."
if zip -r "$ZIP_FILE" "$PAPER_DIR"; then
    log "Zip file created successfully."
else
    log "Failed to create zip file."
    exit 1
fi

# Debug step
log "Sending debug request to $DEBUG_URL."
if curl -s -X POST "$DEBUG_URL" \
    -F "project=@$ZIP_FILE" \
    -F "main=$MAIN_FILE" | jq .; then
    log "Debug request completed successfully."
else
    log "Debug request failed."
    exit 1
fi

# Compile step
log "Sending compile request to $COMPILE_URL."
if curl -s -X POST "$COMPILE_URL" \
    -F "project=@$ZIP_FILE" \
    -F "main=$MAIN_FILE" \
    -F "engine=$ENGINE" \
    -F "jobname=main" \
    -o "$OUTPUT_FILE" -D -; then
    log "Compile request completed successfully."
else
    log "Compile request failed."
    exit 1
fi

# Check if output file exists
if [ -f "$OUTPUT_FILE" ]; then
    log "Compilation succeeded. Opening $OUTPUT_FILE in Firefox."
    firefox "$OUTPUT_FILE" &
else
    log "Compilation failed. Output file not found."
    exit 1
fi

log "Paper compilation process completed."
rm -f "$ZIP_FILE"
log "Cleaned up zip file."