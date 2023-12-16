import torch
import torch.nn as nn
import torch.nn.functional as F
from .Embed import DataEmbedding_wo_pos_tem
from .AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from bayesian_torch.layers import LinearReparameterization as linear_var, Conv1dReparameterization as conv1d_var
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos_tem(configs.enc_in, 
                                                      configs.d_model, 
                                                      configs.embed, 
                                                      configs.freq,
                                                      configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, 
                                        attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # self.fc_out = linear_var(configs.d_model, configs.c_out)
        self.conv_out_s = conv1d_var(configs.seq_len, configs.seq_len, 1, stride=1)
        self.conv_out_t = conv1d_var(configs.seq_len, configs.seq_len, 1, stride=1)
        self.conv_out_t_s = conv1d_var(configs.seq_len, configs.pred_len, 1, stride=1)

    def forward(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_s, enc_t, attns = self.encoder(enc_out, attn_mask=None)

        x_s, kl1 = self.conv_out_s(enc_s)
        x_t, kl2 = self.conv_out_s(enc_t)
        x, kl3 = self.conv_out_t_s(x_s + x_t) 

        kl = kl1+kl2+kl3

        return x,kl/x_enc.size(0)