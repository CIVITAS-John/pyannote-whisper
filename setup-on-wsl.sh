sudo apt update && sudo apt install python3 && sudo apt install ffmpeg
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
pip install requirements.txt