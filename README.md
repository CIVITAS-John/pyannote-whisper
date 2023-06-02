# pyannote-whisper
For more detailed documentation, look for the upstream repository. This repository provides very simple shell scripts to help people without any ML experiences to start using OpenAI Whisper (the AI for transcribing) and/or pyannote (the AI to make speaker labels).

## Setup everything
* You need to first register a Huggingface account.
* Then, generate a [Huggingface Token](https://huggingface.co/settings/tokens). Click `New token` and set the role to `Read`.

### Windows
Throughout the process, we will be in the Windows Subsystem of Linux (WSL) version 2.
* First, install WSL2 with [Ubuntu 22.04](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support).
* Then, enable CUDA on WSL2 following the [link](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).
* Then, download and unzip the folder by running `wget https://github.com/CIVITAS-John/pyannote-whisper/archive/refs/heads/main.zip && unzip main.zip && cd pyannote-whisper-main`.
* Then, run `sh ./setup-on-wsl.sh` to finish the setup on WSL (probably for Ubuntu as well).
* Now you should be all set!

### macOS
* Download and unzip [this repository](https://github.com/CIVITAS-John/pyannote-whisper/archive/refs/heads/main.zip).
* Enter the unzipped folder. Run `sh ./setup-on-mac.sh`. You should be all set.

## Convert your video to audio
Open the terminal, run:
`sh ./extract-audio.sh my_video`
This will convert `my_video.mp4` (Video) to `my_video.ogg` (Audio) in the working folder.

## Convert your audio to transcript with speaker labels
Open the terminal, run:
`sh ./transcribe.sh PATH_TO_YOUR_MP4 {YOUR_HUGGINGFACE_TOKEN}`
This will transcribe `my_video.ogg` (Audio) to `my_video.vtt` (Transcript with speaker and time labels) and my_video.txt (Plain transcript). If you already know the number of speakers (e.g. in one-on-one interviews, 2), you can add the number at the end of the command. For example:
`sh ./transcribe.sh PATH_TO_YOUR_MP4 {YOUR_HUGGINGFACE_TOKEN} 2`

## How to make the model faster
By default, the `transcribe.sh` uses the `medium.en` model, which asks for 5GB of memory and works not really fast. If you are under resource or time constraints, try to change to `tiny.en` (1GB) or `base.en` (1GB). Note that there is also multilingual models, simply remove `.en` to `tiny`, `base`, `small`, `medium`, or `large`.

## Credits
For academic researchers:
* Please acknowledge the author of the helper script (@CIVITAS-John).
* Please cite the following models:
1. [Whisper](https://github.com/openai/whisper): Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust speech recognition via large-scale weak supervision. arXiv preprint arXiv:2212.04356.
1. [pyannote](https://huggingface.co/pyannote/speaker-diarization): Bredin, H., Yin, R., Coria, J. M., Gelly, G., Korshunov, P., Lavechin, M., ... & Gill, M. P. (2020, May). Pyannote. audio: neural building blocks for speaker diarization. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 7124-7128). IEEE.