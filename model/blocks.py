import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from fairseq.modules import LightweightConv


class SCLN(nn.Module):
    """ Speaker Condition Layer Normalization """

    def __init__(self, s_size, hidden_size, eps=1e-8, bias=False):
        super(SCLN, self).__init__()
        self.hidden_size = hidden_size
        self.affine_layer = LinearNorm(
            s_size,
            2 * hidden_size,  # For both b (bias) and g (gain)
            bias,
        )
        self.eps = eps

    def forward(self, x, s):

        # Normalize Input Features
        mu, sigma = torch.mean(
            x, dim=-1, keepdim=True), torch.std(x, dim=-1, keepdim=True)
        y = (x - mu) / (sigma + self.eps)  # [B, T, H_m]

        # Get Bias and Gain
        # [B, 1, 2 * H_m] --> 2 * [B, 1, H_m]
        b, g = torch.split(self.affine_layer(s), self.hidden_size, dim=-1)

        # Perform Scailing and Shifting
        o = g * y + b  # [B, T, H_m]

        return o


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x


class LConvBlock(nn.Module):
    """ Lightweight Convolutional Block """

    def __init__(self, d_model, kernel_size, num_heads, dropout, weight_softmax=True):
        super(LConvBlock, self).__init__()
        self.embed_dim = d_model
        padding_l = (
            kernel_size // 2
            if kernel_size % 2 == 1
            else ((kernel_size - 1) // 2, kernel_size // 2)
        )

        self.act_linear = LinearNorm(
            self.embed_dim, 2 * self.embed_dim, bias=True)
        self.act = nn.GLU()

        self.conv_layer = LightweightConv(
            self.embed_dim,
            kernel_size,
            padding_l=padding_l,
            weight_softmax=weight_softmax,
            num_heads=num_heads,
            weight_dropout=dropout,
        )

        self.fc1 = LinearNorm(self.embed_dim, 4 * self.embed_dim, bias=True)
        self.fc2 = LinearNorm(4 * self.embed_dim, self.embed_dim, bias=True)
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, mask=None):

        x = x.contiguous().transpose(0, 1)

        residual = x
        x = self.act_linear(x)
        x = self.act(x)
        if mask is not None:
            x = x.masked_fill(mask.transpose(0, 1).unsqueeze(2), 0)
        x = self.conv_layer(x)
        x = residual + x

        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        x = x.contiguous().transpose(0, 1)
        x = self.layer_norm(x)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(2), 0)

        return x


class SCLNLConvBlock(nn.Module):
    """ SCLN-LConv Block """

    def __init__(self, d_model, s_size, kernel_size, num_heads, dropout, weight_softmax=True):
        super(SCLNLConvBlock, self).__init__()
        self.embed_dim = d_model
        padding_l = (
            kernel_size // 2
            if kernel_size % 2 == 1
            else ((kernel_size - 1) // 2, kernel_size // 2)
        )

        self.act_linear = LinearNorm(
            self.embed_dim, 2 * self.embed_dim, bias=True)
        self.act = nn.GLU()

        self.conv_layer = LightweightConv(
            self.embed_dim,
            kernel_size,
            padding_l=padding_l,
            weight_softmax=weight_softmax,
            num_heads=num_heads,
            weight_dropout=dropout,
        )
        self.SCLN_1 = SCLN(s_size, self.embed_dim)

        self.fc1 = LinearNorm(self.embed_dim, 4 * self.embed_dim, bias=True)
        self.fc2 = LinearNorm(4 * self.embed_dim, self.embed_dim, bias=True)
        self.SCLN_2 = SCLN(s_size, self.embed_dim)

    def forward(self, x, speaker_embed, mask=None):

        x = x.contiguous().transpose(0, 1)
        speaker_embed = speaker_embed.unsqueeze(1)

        residual = x
        x = self.act_linear(x)
        x = self.act(x)
        if mask is not None:
            x = x.masked_fill(mask.transpose(0, 1).unsqueeze(2), 0)
        x = self.conv_layer(x)
        x = residual + x
        x = self.SCLN_1(x.contiguous().transpose(0, 1),
                        speaker_embed).contiguous().transpose(0, 1)

        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        x = x.contiguous().transpose(0, 1)
        x = self.SCLN_2(x, speaker_embed)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(2), 0)

        return x


class ConvBlock(nn.Module):
    """ Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, dropout, activation=nn.ReLU()):
        super(ConvBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="tanh",
            ),
            nn.BatchNorm1d(out_channels),
            activation
        )
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, enc_input, mask=None):
        enc_output = enc_input.contiguous().transpose(1, 2)
        enc_output = F.dropout(self.conv_layer(
            enc_output), self.dropout, self.training)

        enc_output = self.layer_norm(enc_output.contiguous().transpose(1, 2))
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        transpose=False,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.transpose = transpose

    def forward(self, x):
        if self.transpose:
            x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        if self.transpose:
            x = x.contiguous().transpose(1, 2)

        return x


class FFTBlock(nn.Module):
    """ FFT Block """

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = LinearNorm(d_model, n_head * d_k)
        self.w_ks = LinearNorm(d_model, n_head * d_k)
        self.w_vs = LinearNorm(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = LinearNorm(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output
