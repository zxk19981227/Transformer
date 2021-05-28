import math
from torch.nn.functional import softmax
import torch
from torch.nn import Module, Linear, Dropout
from torch import Tensor


class MultiHeadAttention(Module):
    def __init__(self, num_heads: int, features_dim: int, dropout: float):
        """

        :param num_heads: head num of attention
        :param features_dim: total features dimension4
        """
        super().__init__()
        assert features_dim % num_heads == 0
        hidden_size = features_dim // num_heads
        self.features = features_dim
        self.q_linear = Linear(features_dim, features_dim)
        self.k_linear = Linear(features_dim, features_dim)
        self.v_linear = Linear(features_dim, features_dim)
        self.Dropout = Dropout(dropout)
        self.Dropout2 = Dropout(dropout)
        self.Dropout3 = Dropout(dropout)
        self.dense_linear = Linear(features_dim, features_dim)
        self.dk = math.sqrt(hidden_size)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

    def ScaledDotProduct(self, Q: Tensor, K: Tensor, V: Tensor, mask=None
                         ) -> Tensor:
        """

        :param Q: query vectors ,shape(bzs,num_head,seq_len_q,hidden_size)
        :param K: key vectors ,shape (bzs,num_head,seq_len_k,hidden_size)
        :param V: value vectors ,shape(bzs,num_head,seq_len_v,hidden_size)
        :param mask : mask vector, type bool, shape(bzs,seq_lenq,seq_len_k)
        :return: ask_value ,shape(bzs,num_head,seq_len,hidden_size)
        """
        dk=math.sqrt(K.shape[-1])
        QK = torch.matmul(Q, K.transpose(-1, -2)) /dk
        # QK=self.Dropout(QK)
        # mask=mask.unsqueez.e(1).unsqueeze(2)#shape(bzs,1,1,seq_len)
        if mask != None:
            mask = mask.float() * (-9e10)
            # mask=mask.repeat(1,self.num_heads)#shape(bzs,num_heads,seq_len,seq_len)
            # assert mask.shape==QK.shape
            QK = QK + mask
        scal_product = torch.matmul(softmax(QK, -1), V)
        # scal_product=self.Dropout2(scal_product)
        return scal_product

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor) -> Tensor:
        """

        :param Q: original query vector, shape(bzx,seq_len_q,embedding_size)
        :param K: original query vector, shape(bzx,seq_len_k,embedding_size)
        :param V: original query vector, shape(bzx,seq_len_v,embedding_size)
        :param mask: attention mask,shape
        :return: tensor,shape(bzs,seq_len_q,embedding_size
        """
        Q = self.q_linear(Q)
        K = self.k_linear(K)
        V = self.v_linear(V)
        Q = Q.reshape(Q.shape[0], Q.shape[1], self.num_heads, self.hidden_size)
        Q = Q.permute(0, 2, 1, 3)
        K = K.reshape(K.shape[0], K.shape[1], self.num_heads, self.hidden_size)
        K = K.permute(0, 2, 1, 3)
        V = V.reshape(V.shape[0], V.shape[1], self.num_heads, self.hidden_size)
        V = V.permute(0, 2, 1, 3)
        Scaled_Product = self.ScaledDotProduct(Q, K, V, mask)  # shape:bzs,num_head,seq_len_q,hidden_size
        batch_size = Q.shape[0]
        seq_len = Q.shape[2]
        Scaled_Product = self.dense_linear(
            Scaled_Product.permute([0, 2, 1, 3]).reshape(batch_size, seq_len, self.features))
        Scaled_Product = self.Dropout3(Scaled_Product)

        return Scaled_Product


if __name__ == '__main__':
    multihead = MultiHeadAttention(8, 512, 0)
    test_k = torch.tensor([[10, 0, 0],
                           [0, 10, 0],
                           [0, 0, 10],
                           [0, 0, 10]]).float()  # shape(4,3)
    test_v = torch.tensor([[1, 0],
                           [10, 0],
                           [100, 5],
                           [1000, 6]]).float()
    test_q = torch.tensor([[0, 10, 0]]).float()
    print(multihead.ScaledDotProduct(test_q, test_k, test_v, None))
    y = torch.randn(1, 60, 512)
    print(multihead(y, y, y, None).shape)
