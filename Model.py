import torch.nn

from Encoder import Encoder
from Decoder import Decoder
from torch import Tensor,triu,ones
from torch.nn import Module
from torch.nn import Transformer as tf
from torch.nn import TransformerEncoderLayer,TransformerEncoder,TransformerDecoder,TransformerDecoderLayer
from utils import Positional_Encoding
class Transformer(Module):
    def __init__(self,src_vocab_size:int,tgt_vocab_size:int,encoder_layer_num:int,decoder_layer_num:int,hidden_size:int,feedback_size:int,num_head:int,dropout:float,device:str):
        super().__init__()
        self.Encoder=Encoder(src_vocab_size,num_encoder_layer=encoder_layer_num,hidden_size=hidden_size,num_head=num_head,feedward=feedback_size,dropout=dropout,device=device)
        self.Decoder=Decoder(tgt_vocab_size, num_layer=decoder_layer_num,hiddensize=hidden_size,num_head=num_head,feed_back=feedback_size,dropout=dropout,device=device)
        Encoder_layer=TransformerEncoderLayer(nhead=8,d_model=512)
        self.Encoder_off=TransformerEncoder(encoder_layer=Encoder_layer,num_layers=6)
        Decoder_layer=TransformerDecoderLayer(nhead=8,dim_feedforward=2048,d_model=512)
        self.Decoder_off=TransformerDecoder(decoder_layer=Decoder_layer,num_layers=6)
        self.model=tf()
        self.device=device
        self.input_embedding=torch.nn.Embedding(src_vocab_size,512)
        self.output_embedding=torch.nn.Embedding(tgt_vocab_size,512)
        self.positional=Positional_Encoding(512,512,device)
        self.linear=torch.nn.Linear(512,tgt_vocab_size)
    def forward(self,src_input:Tensor,tgt_input:Tensor):
        """

        :param src_input: src word index,shape(bzs,src_seqlen)
        :param tgt_input: target word index,shape(bzs,tgt_seqlen)
        :return: predict tensor ,shape(bzs,tgt_seqlen)
        """

        batch_size=src_input.shape[0]

        src_seq_len=src_input.shape[1]
        src_input_mask=(src_input==0).view(batch_size,1,1,src_seq_len).to(self.device)
        trg_seq_len=tgt_input.shape[1]
        tgt_input_mask=(tgt_input==0).view(batch_size,1,1,trg_seq_len).to(self.device)
        ahead_mask=triu(ones(trg_seq_len,trg_seq_len),1).view(1,1,trg_seq_len,trg_seq_len).bool().cuda()
        # src_input=self.positional(self.input_embedding(src_input)).permute(1,0,2)
        # src_input=self.positional(self.input_embedding(src_input)).permute(1,0,2)
        # decoder_out=self.model(src_input,tgt_input,src_key_padding_mask=src_input_mask.squeeze(),tgt_key_padding_mask=tgt_input_mask.squeeze(),tgt_mask=ahead_mask.squeeze())
        encoder_out=self.Encoder(src_input,src_input_mask)
        # encoder_out=self.Encoder(src_input,src_key_padding_mask=src_input_mask.squeeze())
        # decoder_out=self.Decoder(tgt_input,encoder_out.transpose(1,0),for)
        decoder_out=self.Decoder(tgt_input,encoder_out,src_input_mask,ahead_mask)
        # decoder_out=self.Decoder(tgt_input,encoder_out,src_input_mask,ahead_mask)
        # decoder_out=self.linear(decoder_out)
        return decoder_out

    def forward_en(self,src_input:Tensor,tgt_input:Tensor):
        """
        这里是替换decoder保证encoder不变
        :param src_input:
        :param tgt_input:
        :return:
        """
        batch_size = src_input.shape[0]

        src_seq_len = src_input.shape[1]
        src_input_mask = (src_input == 0).view(batch_size, 1, 1, src_seq_len).to(self.device)
        trg_seq_len = tgt_input.shape[1]
        tgt_input_mask = (tgt_input == 0).view(batch_size, 1, 1, trg_seq_len).to(self.device)
        ahead_mask = triu(ones(trg_seq_len, trg_seq_len), 1).view(1, 1, trg_seq_len, trg_seq_len).bool().to(self.device)
        # src_input=self.positional(self.input_embedding(src_input)).permute(1,0,2)
        tgt_input=self.positional(self.output_embedding(tgt_input)).permute(1,0,2)
        # decoder_out=self.model(src_input,tgt_input,src_key_padding_mask=src_input_mask.squeeze(),tgt_key_padding_mask=tgt_input_mask.squeeze(),tgt_mask=ahead_mask.squeeze())
        encoder_out = self.Encoder(src_input, src_input_mask)
        # encoder_out=self.Encoder(src_input,src_key_padding_mask=src_input_mask.squeeze())
        # decoder_out=self.Decoder(tgt_input,encoder_out.transpose(1,0),for)
        # decoder_out = self.Decoder(tgt_input, encoder_out, src_input_mask, ahead_mask)
        decoder_out=self.Decoder_off(tgt_input,encoder_out.transpose(1,0),memory_key_padding_mask=src_input_mask.squeeze(),tgt_mask=ahead_mask.squeeze())
        decoder_out=self.linear(decoder_out)
        return decoder_out.transpose(1,0)
    def forward_de(self,src_input:Tensor,tgt_input:Tensor):
        """
        这里是替换encoder保证decoder不变
        :param src_input:
        :param tgt_input:
        :return:
        """
        batch_size = src_input.shape[0]

        src_seq_len = src_input.shape[1]
        src_input_mask = (src_input == 0).view(batch_size, 1, 1, src_seq_len).to(self.device)
        trg_seq_len = tgt_input.shape[1]
        tgt_input_mask = (tgt_input == 0).view(batch_size, 1, 1, trg_seq_len).to(self.device)
        ahead_mask = triu(ones(trg_seq_len, trg_seq_len), 1).view(1, 1, trg_seq_len, trg_seq_len).bool().to(self.device)
        # src_input=self.positional(self.input_embedding(src_input)).permute(1,0,2)
        src_input=self.positional(self.input_embedding(src_input)).permute(1,0,2)
        # decoder_out=self.model(src_input,tgt_input,src_key_padding_mask=src_input_mask.squeeze(),tgt_key_padding_mask=tgt_input_mask.squeeze(),tgt_mask=ahead_mask.squeeze())
        encoder_out = self.Encoder_off(src_input, src_key_padding_mask=src_input_mask.squeeze())
        # encoder_out=self.Encoder(src_input,src_key_padding_mask=src_input_mask.squeeze())
        # decoder_out=self.Decoder(tgt_input,encoder_out.transpose(1,0),for)
        decoder_out = self.Decoder(tgt_input, encoder_out.transpose(1,0),attention_mask=src_input_mask,ahead_mask=ahead_mask)
        # decoder_out=self.Decoder_off(tgt_input,encoder_out,memory_key_padding_mask=src_input_mask.squeeze(),tgt_mask=ahead_mask.squeeze())
        # decoder_out=self.linear(decoder_out)
        return decoder_out
    def forward_off(self,src_input:Tensor,tgt_input:Tensor):
        """
               这里是替换整个transformer
               :param src_input:
               :param tgt_input:
               :return:
               """
        batch_size = src_input.shape[0]

        src_seq_len = src_input.shape[1]
        src_input_mask = (src_input == 0).view(batch_size, 1, 1, src_seq_len).to(self.device)
        trg_seq_len = tgt_input.shape[1]
        tgt_input_mask = (tgt_input == 0).view(batch_size, 1, 1, trg_seq_len).to(self.device)
        ahead_mask = triu(ones(trg_seq_len, trg_seq_len), 1).view(1, 1, trg_seq_len, trg_seq_len).bool().to(self.device)
        # src_input=self.positional(self.input_embedding(src_input)).permute(1,0,2)
        src_input = self.positional(self.input_embedding(src_input)).permute(1, 0, 2)
        tgt_input=self.positional(self.output_embedding(tgt_input)).permute(1,0,2)
        # decoder_out=self.model(src_input,tgt_input,src_key_padding_mask=src_input_mask.squeeze(),tgt_key_padding_mask=tgt_input_mask.squeeze(),tgt_mask=ahead_mask.squeeze())
        encoder_out = self.Encoder_off(src_input, src_key_padding_mask=src_input_mask.squeeze())
        # encoder_out=self.Encoder(src_input,src_key_padding_mask=src_input_mask.squeeze())
        # decoder_out=self.Decoder(tgt_input,encoder_out.transpose(1,0),for)
        # decoder_out = self.Decoder(tgt_input, encoder_out.transpose(1, 0), key_padding_mask=src_input_mask, attn_mask=ahead_mask)
        decoder_out=self.Decoder_off(tgt_input,encoder_out,memory_key_padding_mask=src_input_mask.squeeze(),tgt_mask=ahead_mask.squeeze())
        decoder_out=self.linear(decoder_out)
        return decoder_out.transpose(1,0)