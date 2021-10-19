import os
import json

import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad

from .blocks import (
    LinearNorm,
    LConvBlock,
    SCLNLConvBlock,
    FFTBlock,
)
from text.symbols import symbols


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class TextEncoder(nn.Module):
    """ Text Encoder """

    def __init__(self, config):
        super(TextEncoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=0
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class GlobalEmotionToken(nn.Module):
    """ Global Emotion Token """

    def __init__(self, preprocess_config, model_config):
        super(GlobalEmotionToken, self).__init__()
        self.encoder = ReferenceEncoder(preprocess_config, model_config)
        self.etl = ETL(preprocess_config, model_config)

    def forward(self, inputs, emotions):
        enc_out = None
        if inputs is not None:
            assert emotions is None
            enc_out = self.encoder(inputs)
        else:
            assert emotions is not None
        emotion_embed_hard, emotion_embed_soft, score_hard, score_soft = self.etl(
            enc_out, emotions)

        return emotion_embed_hard, emotion_embed_soft, score_hard, score_soft


class ReferenceEncoder(nn.Module):
    """ Reference Mel Encoder """

    def __init__(self, preprocess_config, model_config):
        super(ReferenceEncoder, self).__init__()

        E = model_config["transformer"]["encoder_hidden"]
        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        ref_enc_filters = model_config["emotion_token_layer"]["ref_enc_filters"]
        ref_enc_size = model_config["emotion_token_layer"]["ref_enc_size"]
        ref_enc_strides = model_config["emotion_token_layer"]["ref_enc_strides"]
        ref_enc_pad = model_config["emotion_token_layer"]["ref_enc_pad"]
        ref_enc_gru_size = model_config["emotion_token_layer"]["ref_enc_gru_size"]

        self.n_mel_channels = n_mel_channels
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=ref_enc_size,
                           stride=ref_enc_strides,
                           padding=ref_enc_pad) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_channels,
                          hidden_size=E // 2,
                          batch_first=True)

    def forward(self, inputs):
        """
        inputs --- [N, Ty/r, n_mels*r]
        outputs --- [N, ref_enc_gru_size]
        """
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.n_mel_channels)  # [N, 1, Ty, n_mels]
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, E//2]

        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class ETL(nn.Module):
    """ Emotion Token Layer """

    def __init__(self, preprocess_config, model_config):
        super(ETL, self).__init__()

        E = model_config["transformer"]["encoder_hidden"]
        num_heads = 1  # model_config["emotion_token_layer"]["num_heads"]
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "emotions.json"
            ),
            "r",
        ) as f:
            token_num = len(json.load(f))

        self.token_num = token_num
        self.embed = nn.Parameter(torch.FloatTensor(
            token_num, E // num_heads))
        d_q = E // 2
        d_k = E // num_heads
        self.attention = StyleEmbedAttention(
            query_dim=d_q, key_dim=d_k, num_units=E, num_heads=num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs, score_hard=None):
        if inputs is not None:
            N = inputs.size(0)
            query = inputs.unsqueeze(1)  # [N, 1, E//2]
        else:
            N = score_hard.size(0)
            query = None
        keys_soft = torch.tanh(self.embed).unsqueeze(0).expand(
            N, -1, -1)  # [N, token_num, E // num_heads]
        if score_hard is not None:
            score_hard = F.one_hot(
                score_hard, self.token_num).float().detach()  # [N, token_num]
        emotion_embed_hard, emotion_embed_soft, score_soft = self.attention(
            query, keys_soft, score_hard)

        return emotion_embed_hard, emotion_embed_soft, score_hard, score_soft.squeeze(0).squeeze(1) if score_soft is not None else None


class StyleEmbedAttention(nn.Module):
    """ StyleEmbedAttention """

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super(StyleEmbedAttention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(
            in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim,
                               out_features=num_units, bias=False)
        self.W_value = nn.Linear(
            in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key_soft, score_hard=None):
        """
        input:
            query --- [N, T_q, query_dim]
            key_soft --- [N, T_k, key_dim]
            score_hard --- [N, T_k]
        output:
            out --- [N, T_q, num_units]
        """
        values = self.W_value(key_soft)
        split_size = self.num_units // self.num_heads
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)

        if query is not None:
            querys = self.W_query(query)  # [N, T_q, num_units]
            keys = self.W_key(key_soft)  # [N, T_k, num_units]

            # [h, N, T_q, num_units/h]
            querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
            # [h, N, T_k, num_units/h]
            keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
            # [h, N, T_k, num_units/h]

            # score = softmax(QK^T / (d_k ** 0.5))
            scores_soft = torch.matmul(
                querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
            scores_soft = scores_soft / (self.key_dim ** 0.5)
            scores_soft = F.softmax(scores_soft, dim=3)

            # out = score * V
            # [h, N, T_q, num_units/h]
            out_soft = torch.matmul(scores_soft, values)
            out_soft = torch.cat(torch.split(out_soft, 1, dim=0), dim=3).squeeze(
                0)  # [N, T_q, num_units]
            out_hard = None
        if score_hard is not None:
            # [N, T_k] -> [h, N, T_q, T_k]
            score_hard = score_hard.unsqueeze(0).unsqueeze(2).repeat(
                self.num_heads, 1, 1, 1)
            out_hard = torch.matmul(score_hard, values)
            out_hard = torch.cat(torch.split(out_hard, 1, dim=0), dim=3).squeeze(
                0)  # [N, T_q, num_units]
            out_soft = scores_soft = None

        return out_hard, out_soft, scores_soft


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, config):
        super(DurationPredictor, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["duration_predictor_hidden"]
        n_layers = config["transformer"]["duration_predictor_layer"]
        n_head = config["transformer"]["duration_predictor_head"]
        d_k = d_v = (
            config["transformer"]["duration_predictor_hidden"]
            // config["transformer"]["duration_predictor_head"]
        )
        d_model = config["transformer"]["duration_predictor_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        lkernel_size = config["transformer"]["lconv_kernel_size"]
        dropout = config["transformer"]["duration_predictor_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.lconv_stack = nn.ModuleList(
            [
                LConvBlock(
                    d_model, lkernel_size, n_head, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.linear_layer = LinearNorm(d_model, 1)

    def forward(self, x, mask, speaker_embed, return_attns=False):

        slf_attn_list = []
        batch_size, max_len = x.shape[0], x.shape[1]

        if not self.training and x.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            out = x + get_sinusoid_encoding_table(
                x.shape[1], self.d_model
            )[: x.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                x.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            out = x[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        residual = out.clone()

        for layer in self.layer_stack:
            out, slf_attn = layer(
                out, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                slf_attn_list += [slf_attn]

        fused_out = out.clone()

        out = out + residual + speaker_embed.unsqueeze(1).expand(
            -1, max_len, -1
        )
        for layer in self.lconv_stack:
            out = layer(
                out, mask=mask
            )
        dur_out = self.linear_layer(out)

        return fused_out, dur_out, mask


class Upsampling(nn.Module):
    """ Upsampling """

    def __init__(self):
        super(Upsampling, self).__init__()
        self.length_regulator = LengthRegulator()

    def forward(
        self,
        x,
        log_duration_prediction,
        mel_mask=None,
        max_len=None,
        duration_target=None,
    ):
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                # * d_control
                torch.round(torch.exp(log_duration_prediction) - 1),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        return (
            x,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class Decoder(nn.Module):
    """ Spectrogram Decoder With Iterative Mel Prediction """

    def __init__(self, preprocess_config, model_config):
        super(Decoder, self).__init__()

        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        n_layers = model_config["decoder"]["decoder_layer"]
        n_layers = model_config["decoder"]["decoder_layer"]
        n_head = model_config["decoder"]["decoder_head"]
        d_model = model_config["decoder"]["decoder_hidden"]
        kernel_size = model_config["decoder"]["conv_kernel_size"]
        dropout = model_config["decoder"]["decoder_dropout"]

        self.max_seq_len = model_config["max_seq_len"]
        self.d_model = d_model
        self.n_layers = n_layers

        self.lconv_stack = nn.ModuleList(
            [
                SCLNLConvBlock(
                    d_model, d_model, kernel_size, n_head, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.mel_projection = nn.ModuleList(
            [
                LinearNorm(
                    d_model, n_mel_channels
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, speaker_embed, mask):
        mel_iters = []
        out = x
        for i, (conv, linear) in enumerate(zip(self.lconv_stack, self.mel_projection)):
            out = out.masked_fill(mask.unsqueeze(-1), 0)
            out = conv(
                out, speaker_embed, mask=mask
            )
            if self.training or not self.training and i == self.n_layers-1:
                mel_iters.append(
                    linear(out).masked_fill(mask.unsqueeze(-1), 0)
                )
        return mel_iters
