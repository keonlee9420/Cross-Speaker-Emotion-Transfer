import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

intensity_dict = {
    "01": "normal",
    "02": "strong",
}

script_dict = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}


def parse_filename(filename, speaker):
    res = []
    name_list = filename.split("-")
    res.append(emotion_dict[name_list[2]])
    res.append(intensity_dict[name_list[3]])
    res.append(name_list[5])
    res.append(speaker)
    return "-".join(res), script_dict[name_list[4]]


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    for spker_id, speaker in enumerate(tqdm(os.listdir(in_dir))):
        if "Actor_" not in speaker:
            continue
        for i, wav_name in enumerate(tqdm(os.listdir(os.path.join(in_dir, speaker)))):
            wav_path = os.path.join(os.path.join(in_dir, speaker, wav_name))
            base_name, text = parse_filename(wav_name.split(".")[0], speaker)
            text = _clean_text(text, cleaners)

            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker,
                                 "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, speaker,
                                 "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)
            else:
                print("[Error] No flac file:{}".format(wav_path))
                continue
