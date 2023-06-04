#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <directory> [transcribe.sh arguments...]"
  exit 1
fi

# Define color variables
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'  # No color

# Find all .mp4 files
directory="$1"
shift

find "$directory" -type f -name '*.mp4' | while IFS= read -r file; do
  file="${file%.mp4}"  # Remove the .mp4 extension from the file name
  
  vtt_file="$file.vtt"  # VTT file path

  # Check if .srt file exists for the current video file
  if [ -e "$vtt_file" ]; then
    echo "${YELLOW}Skipping: $file (VTT file already exists)${NC}"
  else
    echo "${YELLOW}Converting: $file${NC}"
    bash extract-audio.sh "$file" $@ </dev/null
    echo "${GREEN}Transcribing: $file${NC}"
    bash transcribe.sh "$file" $@ </dev/null
    echo "${GREEN}Finished transcribing: $file${NC}"
  fi
done