import re
import os
import json
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
# from pypinyin import pinyin, Style
import tgt
import librosa
import audio as Audio

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, synth_samples, get_alignment
from dataset import TextDataset
from text import text_to_sequence


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def get_audio(preprocess_config, wav_path):

    preprocessed_path = preprocess_config["path"]["preprocessed_path"]
    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    STFT = Audio.stft.TacotronSTFT(
        preprocess_config["preprocessing"]["stft"]["filter_length"],
        hop_length,
        preprocess_config["preprocessing"]["stft"]["win_length"],
        preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        sampling_rate,
        preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )

    # Check TextGrid
    basename = wav_path.split("/")[-1].replace(".wav", "")
    speaker = basename.split("-")[-1]
    tg_path = os.path.join(preprocessed_path, "TextGrid",
                           speaker, f'{basename}.TextGrid')
    if os.path.exists(tg_path):
        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        _, duration, start, end = get_alignment(
            textgrid.get_tier_by_name("phones"),
            sampling_rate,
            hop_length,
        )
        if start >= end:
            raise ValueError()

    # Read and trim wav files
    wav, _ = librosa.load(wav_path, sampling_rate)
    if os.path.exists(tg_path):
        wav = wav[
            int(sampling_rate * start): int(sampling_rate * end)
        ]

    # Compute mel-scale spectrogram
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(
        wav.astype(np.float32), STFT)
    if os.path.exists(tg_path):
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]

    mels = mel_spectrogram.T[None].astype(np.float32)
    mel_lens = np.array([len(mels[0])])

    return mels, mel_lens


def synthesize(device, model, args, configs, vocoder, batchs):
    preprocess_config, model_config, train_config = configs

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(*batch[2:])
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                args,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default="Actor_01",
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    parser.add_argument(
        "--emotion_id",
        type=str,
        default=None,
        help="emotion ID for multi-emotion synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="reference audio path to extract the speech style, for single-sentence mode only",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(
        args.dataset)
    configs = (preprocess_config, model_config, train_config)
    os.makedirs(
        os.path.join(train_config["path"]["result_path"], str(args.restore_step)), exist_ok=True)

    # Set Device
    torch.manual_seed(train_config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(train_config["seed"])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device of XSpkEmoTrans:", device)

    # Get model
    model = get_model(args, configs, device, train=False,
                      ignore_layers=train_config["ignore_layers"])

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config, model_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]

        # Speaker Info
        load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        speakers = np.array([speaker_map[args.speaker_id]]) if model_config["multi_speaker"] else np.array(
            [0])  # single speaker is allocated 0
        spker_embed = np.load(os.path.join(
            preprocess_config["path"]["preprocessed_path"],
            "spker_embed",
            "{}-spker_embed.npy".format(args.speaker_id),
        )) if load_spker_embed else None

        # Emotion Info
        emotions = None
        if args.emotion_id is not None:
            with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json")) as f:
                emotions = np.array([json.load(f)[args.emotion_id]])
        mels = mel_lens = None
        if args.ref_audio is not None:
            mels, mel_lens = get_audio(preprocess_config, args.ref_audio)
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array(
                [preprocess_english(args.text, preprocess_config)])
        else:
            raise NotImplementedError()
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(
            text_lens), mels, mel_lens, max(mel_lens) if mels is not None else None, spker_embed, emotions)]

    synthesize(device, model, args, configs, vocoder, batchs)
