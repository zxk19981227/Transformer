from Encoder import Encoder
from Decoder import Decoder
from torch import Tensor,triu,ones
from torch.nn import Module
class Transformer(Module):
    def __init__(self,src_vocab_size:int,tgt_vocab_size:int,encoder_layer_num:int,decoder_layer_num:int,hidden_size:int,feedback_size:int,num_head:int,dropout:float,device:str):
        super().__init__()
        self.Encoder=Encoder(src_vocab_size,num_encoder_layer=encoder_layer_num,hidden_size=hidden_size,num_head=num_head,feedward=feedback_size,dropout=dropout,device=device)
        self.Decoder=Decoder(tgt_vocab_size, num_layer=decoder_layer_num,hiddensize=hidden_size,num_head=num_head,feed_back=feedback_size,dropout=dropout,device=device)
        self.device=device

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
        ahead_mask=triu(ones(trg_seq_len,trg_seq_len),1).view(1,1,trg_seq_len,trg_seq_len).cuda()
        encoder_out=self.Encoder(src_input,src_input_mask)
        decoder_out=self.Decoder(tgt_input,encoder_out,src_input_mask,ahead_mask)

        return decoder_out