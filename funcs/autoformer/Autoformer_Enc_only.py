import torch
import torch.nn as nn
from .Embed import DataEmbedding_wo_pos_tem
from .AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .Autoformer_EncDec import Encoder_only, my_Layernorm, series_decomp, EncoderLayer_only
from bayesian_torch.layers import Conv1dReparameterization as conv1d_var
from torch.nn import Conv1d


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
        self.encoder = Encoder_only(
            [
                EncoderLayer_only(
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

    def forward(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_s, enc_t, attns = self.encoder(enc_out, attn_mask=None)
        
        return enc_s, enc_t
    
class Small_decoder_bayes(nn.Module):
    """
    Small Decoder separatly to run multiple trainings and approximate the elbo
    """
    def __init__(self, configs):
        super(Small_decoder_bayes, self).__init__()
        self.conv_out_s = conv1d_var(configs.seq_len, configs.seq_len, 1, stride=1)
        self.conv_out_t = conv1d_var(configs.seq_len, configs.seq_len, 1, stride=1)
        self.conv_out_t_s = conv1d_var(configs.seq_len, configs.pred_len, 1, stride=1)
    def forward(self,enc_s,enc_t):
        x_s, kl1 = self.conv_out_s(enc_s)
        x_t, kl2 = self.conv_out_s(enc_t)
        x, kl3 = self.conv_out_t_s(x_s + x_t) 
        kl = kl1+kl2+kl3
        return x, kl
    
class Small_decoder_freq(nn.Module):
    """
    Small Decoder separatly to run multiple trainings and approximate the elbo
    """
    def __init__(self, configs):
        super(Small_decoder_freq, self).__init__()
        self.conv_out_s = Conv1d(configs.seq_len, configs.seq_len, 1, stride=1)
        self.conv_out_t = Conv1d(configs.seq_len, configs.seq_len, 1, stride=1)
        self.conv_out_t_s = Conv1d(configs.seq_len, configs.pred_len, 1, stride=1)
    def forward(self,enc_s,enc_t):
        x_s = self.conv_out_s(enc_s)
        x_t = self.conv_out_s(enc_t)
        x = self.conv_out_t_s(x_s + x_t) 
        return x