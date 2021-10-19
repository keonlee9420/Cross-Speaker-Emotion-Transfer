import os
import re
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path

import audio as Audio
from model import PreDefinedEmbedder
from utils.tools import plot_embedding, get_alignment


class Preprocessor:
    def __init__(self, preprocess_config, model_config, train_config):
        random.seed(train_config['seed'])
        self.preprocess_config = preprocess_config
        self.multi_speaker = model_config["multi_speaker"]
        self.corpus_dir = preprocess_config["path"]["corpus_path"]
        self.in_dir = preprocess_config["path"]["raw_path"]
        self.out_dir = preprocess_config["path"]["preprocessed_path"]
        self.val_size = preprocess_config["preprocessing"]["val_size"]
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]

        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )
        self.val_prior = self.val_prior_names(
            os.path.join(self.out_dir, "val.txt"))
        self.speaker_emb = None
        self.in_sub_dirs = [p for p in os.listdir(
            self.in_dir) if os.path.isdir(os.path.join(self.in_dir, p))]
        if self.multi_speaker and preprocess_config["preprocessing"]["speaker_embedder"] != "none":
            self.speaker_emb = PreDefinedEmbedder(preprocess_config)
            self.speaker_emb_dict = self._init_spker_embeds(self.in_sub_dirs)

    def _init_spker_embeds(self, spkers):
        spker_embeds = dict()
        for spker in spkers:
            spker_embeds[spker] = list()
        return spker_embeds

    def val_prior_names(self, val_prior_path):
        val_prior_names = set()
        if os.path.isfile(val_prior_path):
            print("Load pre-defined validation set...")
            with open(val_prior_path, "r", encoding="utf-8") as f:
                for m in f.readlines():
                    val_prior_names.add(m.split("|")[0])
            return list(val_prior_names)
        else:
            return None

    def build_from_path(self):
        embedding_dir = os.path.join(self.out_dir, "spker_embed")
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs(embedding_dir, exist_ok=True)

        print("Processing Data ...")
        out = list()
        train = list()
        val = list()
        n_frames = 0
        max_seq_len = -float('inf')

        skip_speakers = set()
        for embedding_name in os.listdir(embedding_dir):
            skip_speakers.add(embedding_name.split("-")[0])

        # Compute duration, and mel-spectrogram
        speakers, emotions, emotion_set = {}, {}, set()
        for i, speaker in enumerate(tqdm(self.in_sub_dirs)):
            save_speaker_emb = self.speaker_emb is not None and speaker not in skip_speakers
            if os.path.isdir(os.path.join(self.in_dir, speaker)):
                speakers[speaker] = i
                for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
                    if ".wav" not in wav_name:
                        continue

                    basename = wav_name.split(".")[0]
                    emotion_set.add(basename.split("-")[0])
                    tg_path = os.path.join(
                        self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(
                            basename)
                    )
                    if os.path.exists(tg_path):
                        ret = self.process_utterance(
                            tg_path, speaker, basename, save_speaker_emb)
                        if ret is None:
                            continue
                        else:
                            info, n, spker_embed = ret

                        if self.val_prior is not None:
                            if basename not in self.val_prior:
                                train.append(info)
                            else:
                                val.append(info)
                        else:
                            out.append(info)
                    if save_speaker_emb:
                        self.speaker_emb_dict[speaker].append(spker_embed)

                    if n > max_seq_len:
                        max_seq_len = n

                    n_frames += n

                # Calculate and save mean speaker embedding of this speaker
                if save_speaker_emb:
                    spker_embed_filename = '{}-spker_embed.npy'.format(speaker)
                    np.save(os.path.join(self.out_dir, 'spker_embed', spker_embed_filename),
                            np.mean(self.speaker_emb_dict[speaker], axis=0), allow_pickle=False)

        for i, e in enumerate(emotion_set):
            emotions[e] = i

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "emotions.json"), "w") as f:
            f.write(json.dumps(emotions))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "max_seq_len": max_seq_len
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        if self.speaker_emb is not None:
            print("Plot speaker embedding...")
            plot_embedding(
                self.out_dir, *self.load_embedding(embedding_dir),
                self.divide_speaker_by_gender(self.out_dir), filename="spker_embed_tsne.png"
            )

        if self.val_prior is not None:
            assert len(out) == 0
            random.shuffle(train)
            train = [r for r in train if r is not None]
            val = [r for r in val if r is not None]
        else:
            assert len(train) == 0 and len(val) == 0
            random.shuffle(out)
            out = [r for r in out if r is not None]
            train = out[self.val_size:]
            val = out[: self.val_size]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in train:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in val:
                f.write(m + "\n")

        return out

    def process_utterance(self, tg_path, speaker, basename, save_speaker_emb):
        wav_path = os.path.join(self.in_dir, speaker,
                                "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker,
                                 "{}.lab".format(basename))

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = get_alignment(
            textgrid.get_tier_by_name("phones"),
            self.sampling_rate,
            self.hop_length,
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path, self.sampling_rate)
        wav = wav.astype(np.float32)
        spker_embed = self.speaker_emb(wav) if save_speaker_emb else None
        wav = wav[
            int(self.sampling_rate * start): int(self.sampling_rate * end)
        ]

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute mel-scale spectrogram
        mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            mel_spectrogram.shape[1],
            spker_embed,
        )

    def divide_speaker_by_gender(self, in_dir, speaker_path="speaker-info.txt"):
        speakers = dict()
        with open(os.path.join(in_dir, speaker_path), encoding='utf-8') as f:
            for line in tqdm(f):
                if "ID" in line:
                    continue
                parts = [p.strip()
                         for p in re.sub(' +', ' ', (line.strip())).split(' ')]
                spk_id, gender = parts[0], parts[2]
                speakers[str(spk_id)] = gender
        return speakers

    def load_embedding(self, embedding_dir):
        embedding_path_list = [_ for _ in Path(embedding_dir).rglob('*.npy')]
        embedding = None
        embedding_speaker_id = list()
        # Gather data
        for path in tqdm(embedding_path_list):
            embedding = np.concatenate((embedding, np.load(path)), axis=0) \
                if embedding is not None else np.load(path)
            embedding_speaker_id.append(
                str(str(path).split('/')[-1].split('-')[0]))
        return embedding, embedding_speaker_id
