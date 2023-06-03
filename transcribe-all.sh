#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <directory> [transcribe.sh arguments...]"
  exit 1
fi

directory="$1"
shift

find "$directory" -type f -name '*.mp4' | while IFS= read -r file; do
  file="${file%.mp4}"  # Remove the .mp4 extension from the file name
  echo "Converting: $file"
  bash extract-audio.sh "$file" $@ </dev/null
  echo "Transcribing: $file"
  bash transcribe.sh "$file" $@ </dev/null
  echo "Finished transcribing: $file"
done