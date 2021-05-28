from MultiHeadAttention import MultiHeadAttention
from torch import Tensor
from torch.nn import Module, LayerNorm, ModuleList, Embedding
from FeedForward import FeedForward
from utils import Positional_Encoding
import math


class EncoderLayer(Module):
    """
    This is the Encoder layer in Transformer
    """

    def __init__(self, hidden_size: int, num_heads: int, feedward: int, dropout: float):
        """

        :param hidden_size: hidden dimensions
        :param num_heads: number of multihead
        :param feedword: backward dimensions
        """
        super().__init__()
        self.MultiHead = MultiHeadAttention(num_heads, hidden_size, dropout)
        self.FeedForward = FeedForward(hidden_size, feedward, dropout)
        self.LayerNorm1 = LayerNorm(hidden_size, eps=1e-6)
        self.LayerNorm2 = LayerNorm(hidden_size, eps=1e-6)

    def forward(self, input_feature: Tensor, mask: Tensor) -> Tensor:
        """

        :param input_feature: embeddidng features, shape:(bzs,seq_len,hidden_dim)
        :param mask: attention mask,shape(bzx,seq_len,hidden_dim)
        :return: shape:(bzs,seq_len,hidden_dim)
        """
        features = self.MultiHead(input_feature, input_feature, input_feature, mask)
        features = self.LayerNorm1(features + input_feature)
        output = self.FeedForward(features)
        output = self.LayerNorm2(features + output)
        return output


class Encoder(Module):
    """
    A single Encoder in the Transformer.
    """

    def __init__(self, vocab_size: int, num_encoder_layer: int, hidden_size: int, num_head: int, feedward: int,
                 dropout: float, device: str):
        """

        :param vocab_size: num of word in source language
        :param num_encoder_layer:  number of encoder layer
        :param hidden_size: the hidden size/ embedding size for single word
        :param num_head: the number of multihead
        :param feedward: hidden dimension for feedback
        """
        super().__init__()
        self.Encoder_layers = ModuleList()
        for i in range(num_encoder_layer):
            self.Encoder_layers.append(EncoderLayer(hidden_size, num_head, feedward, dropout))
        self.Embedding = Embedding(vocab_size, hidden_size, padding_idx=0)
        self.Positional_Encoding = Positional_Encoding(hidden_size, 512, device)
        self.d_model = hidden_size

    def forward(self, input_features, input_mask):
        features = self.Positional_Encoding(self.Embedding(input_features))
        for layer in self.Encoder_layers:
            features = layer(features, input_mask)
        return features
if __name__=='__main__':
    import torch
    Encoder_layer=EncoderLayer(512,8,2048,0)
    output=Encoder_layer(torch.randn(64,43,512),mask=None)
    print(output.shape)