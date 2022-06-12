import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import TextEncoder, GlobalEmotionToken, DurationPredictor, Upsampling, Decoder
from utils.tools import get_mask_from_lengths


class XSpkEmoTrans(nn.Module):
    """ Cross-Speaker Emotion Transfer TTS """

    def __init__(self, preprocess_config, model_config):
        super(XSpkEmoTrans, self).__init__()
        self.model_config = model_config

        self.encoder = TextEncoder(model_config)
        self.emotion_emb = GlobalEmotionToken(preprocess_config, model_config)
        self.duratin_predictor = DurationPredictor(model_config)
        self.upsampling = Upsampling()
        self.decoder = Decoder(preprocess_config, model_config)

        self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
        if self.embedder_type == "none":
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
        else:
            self.speaker_emb = nn.Linear(
                model_config["external_speaker_dim"],
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        spker_embeds=None,
        emotions=None,
        d_targets=None,
        inference=False,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        emotion_embed_hard, emotion_embed_soft, score_hard, score_soft = self.emotion_emb(
            mels, emotions)
        if self.training:
            # print("HARD emotion token")
            output = output + emotion_embed_hard.expand(
                -1, max_src_len, -1
            )
        elif emotion_embed_hard is not None:
            # print("HARD emotion token")
            output = output + emotion_embed_hard.expand(
                -1, max_src_len, -1
            )
        else:
            # print("SOFT emotion token")
            output = output + emotion_embed_soft.expand(
                -1, max_src_len, -1
            )

        if self.embedder_type == "none":
            speaker_embed = self.speaker_emb(speakers)
        else:
            assert spker_embeds is not None, "Speaker embedding should not be None"
            speaker_embed = self.speaker_emb(spker_embeds)

        output, log_d_predictions, src_masks = self.duratin_predictor(
            output, src_masks, speaker_embed)

        (
            output,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.upsampling(
            output,
            log_d_predictions,
            mel_masks if not inference else None,
            max_mel_len if not inference else None,
            d_targets,
        )

        mel_iters = self.decoder(output, speaker_embed, mel_masks)

        return (
            mel_iters,
            score_hard,
            score_soft,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
