from FeedForward import FeedForward
from torch import Tensor
from MultiHeadAttention import MultiHeadAttention
from utils import Positional_Encoding
from torch.nn import Module, LayerNorm, Embedding, ModuleList, Linear
from torch import Tensor
import math


class DecoderLayer(Module):
    def __init__(self, num_head, hiddensize: int, feedback: int, dropout: float):
        """

        :param num_head: heads num of multi_head
        :param hiddensize: embedding size for words
        :param feedback: hidden size in feedback
        """
        super().__init__()
        self.MultiHeadInput = MultiHeadAttention(num_head, hiddensize, dropout)
        self.MultiHeadOutput = MultiHeadAttention(num_head, hiddensize, dropout)
        self.LayerNorm1 = LayerNorm(hiddensize,eps=1e-6)
        self.LayerNorm2 = LayerNorm(hiddensize,eps=1e-6)
        self.LayerNorm3 = LayerNorm(hiddensize,eps=1e-6)
        self.feedback = FeedForward(hiddensize, feedback, dropout)
    def forward(self, input_feature, encoder_out, outputmask, forward_mask):
        """

        :param input_feature: input feature,shape:(bzs,seq_len,hiddensize)
        :param encoder_out: the output of encoder,shape(bzs,encoder_seq_len,hiddensize)
        :param outputmask: input feature size,shape(bzs,seq_len)
        :param forward_mask: translate mask is a Upper Triangular Matrix
        :return: result of decoder ,shape (bzs,decoder_seqlen,hiddensize)
        """
        Layerfeatures = self.MultiHeadInput(input_feature, input_feature, input_feature, forward_mask)
        Layerfeatures = self.LayerNorm1(Layerfeatures + input_feature)
        Layerfeatures = self.LayerNorm2(
            Layerfeatures + self.MultiHeadOutput(Layerfeatures, encoder_out, encoder_out, outputmask))
        Layerfeatures = self.LayerNorm3(Layerfeatures + self.feedback(Layerfeatures))
        return Layerfeatures


class Decoder(Module):
    def __init__(self, vocab_size: int, num_layer: int, num_head: int, hiddensize: int, feed_back: int, dropout: float,
                 device: str):
        """
        :param vocab_size: translation target vocab num
        :param num_layer: decoder layer num
        :param num_head: multihead num
        :param hiddensize: embedding dimension
        :param feed_back: hidden dimension in feedback
        """
        super(Decoder, self).__init__()
        self.Positional = Positional_Encoding(hiddensize, 512, device)
        self.Embedding = Embedding(vocab_size, hiddensize, padding_idx=0)
        self.Decoder_Layer = ModuleList()
        for i in range(num_layer):
            self.Decoder_Layer.append(DecoderLayer(num_head, hiddensize, feed_back, dropout))
        self.feedback = FeedForward(hiddensize, feed_back, dropout=dropout)
        self.d_model = hiddensize
        self.Linear = Linear(hiddensize, vocab_size)
        self.Linear.weight = self.Embedding.weight

    def forward(self, inputs: Tensor, encoder_out, attention_mask: Tensor, ahead_mask: Tensor):
        input_features = self.Positional(self.Embedding(inputs) * math.sqrt(self.d_model))
        for layer in self.Decoder_Layer:
            input_features = layer(input_features, encoder_out, attention_mask, ahead_mask)
        return self.Linear(input_features)
