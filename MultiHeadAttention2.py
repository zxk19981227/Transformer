from torch.nn import MultiheadAttention
import torch


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads: int, features_dim: int, dropout: float):
        """

        :param num_heads: head num of attention
        :param features_dim: total features dimension4
        """
        super().__init__()
        self.model = MultiheadAttention(num_heads=num_heads, embed_dim=features_dim, dropout=dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        if attn_mask!=None:
            attn_mask=attn_mask.squeeze()
        if key_padding_mask!=None:
            key_padding_mask=key_padding_mask.squeeze()
        query = query.transpose(1, 0)
        key = key.transpose(1, 0)
        value = value.transpose(1, 0)
        out, attn = self.model(query=query, key=key, value=value, attn_mask=attn_mask,key_padding_mask=key_padding_mask)
        return out.transpose(1, 0)
if __name__=='__main__':
    multihead = MultiHeadAttention(2,4, 0)
    test_k = torch.tensor([[10, 0, 0],
                           [0, 10, 0],
                           [0, 0, 10],
                           [0, 0, 10]]).float()  # shape(4,3)
    test_v = torch.tensor([[1, 0],
                           [10, 0],
                           [100, 5],
                           [1000, 6]]).float()
    test_q = torch.tensor([[0, 10, 0]]).float()
    #expect [[10,0]]
    tmp_q=torch.tensor([[0,0,10]]).float()
    #expect [[550,5.5]]
    # print(multihead.ScaledDotProduct(tmp_q,test_k,test_v))
    y=torch.tensor([i for i in range(240)])
    y = y.view(1, 60, 4).float()
    print(multihead(y, y, y, None).shape)
    print(multihead(y,y,y,None))
