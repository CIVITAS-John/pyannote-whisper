/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install ffmpeg
brew install openai-whisper
pip3 install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
pip3 install -r requirements.txt