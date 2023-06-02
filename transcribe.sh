
if [ -z "$3" ]; then
    num_speakers="None"
else
    num_speakers="$3"
fi

python3 -m pyannote_whisper.cli.transcribe $1.ogg --model medium.en --language en --diarization True --output_format=TXT --hf_token $2 --num_speakers=$num_speakers