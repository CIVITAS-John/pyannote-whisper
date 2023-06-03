
if [ -z "$3" ]; then
    num_speakers="None"
else
    num_speakers="$3"
fi

output_dir="$(dirname "$1")"

python3 -m pyannote_whisper.cli.transcribe "$1.ogg" --output_dir="$output_dir" --model medium.en --language en --diarization True --output_format=VTT --hf_token $2 --num_speakers=$num_speakers