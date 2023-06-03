import argparse
import requests
import json
import os
import warnings
from typing import Literal, cast

import numpy as np
import torch
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.transcribe import transcribe
from whisper.utils import (WriteSRT, WriteTXT, WriteVTT, optional_float,
                           optional_int, str2bool)

import stable_whisper

from pyannote_whisper.utils import diarize_text, write_to_vtt

def send_slack_notification(webhook_url, audio: list):
    message = f"Transcription finished for {audio}"
    data = {'text': message}
    response = requests.post(webhook_url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
    if response.status_code != 200:
        raise ValueError('Request to slack returned an error %s, the response is:\n%s' % (response.status_code, response.text))

def cli():
    from whisper import available_models

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hf_token", type=str, help="Hugging Face Token to use pre-trained versions of pyannote/speaker-diarization")
    parser.add_argument("--slack_webhook", type=str, help="Slack Token to use for sending messages", default="")
    parser.add_argument("audio", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--verbose", type=str2bool, default=True,
                        help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None,
                        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5,
                        help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5,
                        help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None,
                        help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None,
                        help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1",
                        help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None,
                        help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True,
                        help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True,
                        help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2,
                        help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4,
                        help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0,
                        help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6,
                        help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--threads", type=optional_int, default=0,
                        help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    parser.add_argument("--diarization", type=str2bool, default=True,
                        help="whether to perform speaker diarization; True by default")
    parser.add_argument("--num_speakers", type=optional_int, default=None,
                        help="number of speakers if that is already known; None by default")
    parser.add_argument("--output_format", type=str, default="TXT", choices=['TXT', 'VTT', 'SRT'],
                        help="output format; TXT by default")

    args = parser.parse_args().__dict__
    hf_token = args.pop("hf_token")
    slack_webhook = ""
    if "slack_webhook" in args:
        slack_webhook = args.pop("slack_webhook")
    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    device: str = args.pop("device")
    output_format: Literal['TXT', 'VTT', 'SRT'] = args.pop("output_format")
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead.")
        args["language"] = "en"

    temperature = float(args.pop("temperature"))
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)

    # from whisper import load_model
    # model = load_model(model_name, device=device, download_root=model_dir)

    model = stable_whisper.load_model(model_name)

    diarization = args.pop("diarization")
    num_speakers = args.pop("num_speakers")

    audio_path = args.pop("audio")
    result = model.transcribe(audio_path, temperature=temperature, vad=True, **args)
    # result = transcribe(model, audio_path, temperature=temperature, **args)
    audio_basename = os.path.basename(audio_path)

    result.save_as_json(os.path.join(output_dir, audio_basename + '_raw.json'))
    if output_format == "TSV":
        # save TSV
        result.to_tsv(os.path.join(output_dir, audio_basename + '_raw.tsv'), word_level=False)

    elif output_format == "VTT":
        # save VTT
        result.to_srt_vtt(os.path.join(output_dir, audio_basename + '_raw.vtt'), word_level=False)

    elif output_format == "SRT":
        # save SRT
        result.to_srt_vtt(os.path.join(output_dir, audio_basename + '_raw.srt'), word_level=False)

    del model
    torch.cuda.empty_cache()

    if diarization:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                            use_auth_token=f"{hf_token}")
        if num_speakers == None:
            diarization_result = pipeline(audio_path)
        else:
            diarization_result = pipeline(audio_path, num_speakers=num_speakers)
        filepath = os.path.join(output_dir, audio_basename + ".vtt")
        
        word_segments = result.all_words()

        res = diarize_text(word_segments, diarization_result)
        write_to_vtt(res, filepath)

    # Send slack notification when done
    if slack_webhook != "":
        send_slack_notification(slack_webhook, audio_path)

 

if __name__ == '__main__':
    cli()
