# pyannote-whisper
For more detailed documentation, look for the upstream repository. This repository provides very simple shell scripts to help people without any ML experiences to start using OpenAI Whisper (the AI for transcribing) and/or pyannote (the AI to make speaker labels).

## Setup everything
### Windows
Throughout the process, we will be in the Windows Subsystem of Linux (WSL) version 2.
* First, install WSL2 with [https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support](Ubuntu 22.04).
* Then, enable CUDA on WSL2 following the [https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl](link)
* Then, run `sh ./setup-on-wsl.sh` to finish the setup on WSL (probably for Ubuntu as well).
* Now you should be all set!

### macOS
* Run `sh ./setup-on-mac.sh`. You should be all set.

## Convert your video to audio
Open the terminal, run:
`sh ./extract-audio.sh my_video`
This will convert my_video.mp4 (Video) to my_video.ogg (Audio).

## Convert your audio to transcript with speaker labels
Open the terminal, run:
`sh ./transcribe.sh PATH_TO_YOUR_MP4`
This will transcribe my_video.ogg (Audio) to my_video_labeled.txt (Transcript with speaker and time labels) and my_video.txt (Plain transcript).

## How to make the model faster
By default, the `transcribe.sh` uses the `medium.en` model, which asks for 5GB of memory and works not really fast. If you are under resource or time constraints, try to change to `tiny.en` (1GB) or `base.en` (1GB). Note that there is also multilingual models, simply remove `.en` to `tiny`, `base`, `small`, `medium`, or `large`.