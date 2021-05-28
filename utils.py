import math
from torch.nn.functional import cross_entropy
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import argparse


def get_args():
    """
    Function used to get hyperparameter from user
    :return: hyperparameters arg
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--feedback', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--encoder_layer_num', type=int, default=6)
    parser.add_argument('--decoder_layer_num', type=int, default=6)
    parser.add_argument('--src_lang', type=str, default='de')
    parser.add_argument('--tgt_lang', type=str, default='en')
    parser.add_argument('--data_bin', type=str, default='../dataset/bin')
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--epoch', type=int, default=1000)
    args = parser.parse_args()
    return args


class Positional_Encoding(Module):
    def __init__(self, dim_size: int, max_length: int, device: str):
        super().__init__()
        self.Positional = np.zeros((max_length, dim_size))
        for i in range(max_length):
            for j in range(dim_size):
                if j % 2 == 0:
                    self.Positional[i, j] = math.sin(i / math.pow(10000.0, j / dim_size))
                else:
                    self.Positional[i, j] = math.cos(i / math.pow(10000.0, (j - 1) / dim_size))
        self.Positional = torch.tensor(self.Positional).view(1, max_length, dim_size).to(device).float()

    def forward(self, features):
        features = features + self.Positional[:, :features.shape[1], :]
        return features


def cal_accruacy(predict, label):
    mask = (label != 0)
    predict = torch.argmax(predict, -1)
    correct = ((predict == label) & mask).long().sum().item()
    total = mask.long().sum().item()
    return correct, total


def cal_loss(predict, label):
    loss = cross_entropy(predict.reshape(-1, predict.shape[-1]), label.reshape(-1), ignore_index=0)
    return loss


class Optim():
    def __init__(self, optim: torch.optim.Adam, warmup_step, d_model):
        self.optim = optim
        self.step_num = 0
        self.warmup = warmup_step
        self.d_model = math.pow(d_model, -0.5)

    def get_lr(self) -> float:
        self.step_num += 1
        lr = min(math.pow(self.step_num, -0.5), self.step_num * math.pow(self.warmup, -1.5)) * self.d_model
        return lr

    def step(self):
        lr = self.get_lr()
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
        self.optim.step()
    def zero_grad(self):
        self.optim.zero_grad()


def collate_fn(batch):
    src = []
    trg = []
    for s, t in batch:
        src.append(s)
        trg.append(t)
    src = pad_sequence(src, batch_first=True, padding_value=0)
    trg = pad_sequence(trg, batch_first=True, padding_value=0)
    return src, trg


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt

    n, d = 2048, 512
    pos_encoding = Positional_Encoding(d, n, 'cpu')
    print(pos_encoding.Positional.shape)
    pos_encoding = pos_encoding.Positional[0]

    # Juggle the dimensions for the plot
    pos_encoding = torch.reshape(pos_encoding, (n, d // 2, 2))
    pos_encoding = pos_encoding.permute(2, 1, 0)
    pos_encoding = torch.reshape(pos_encoding, (d, n))

    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()

    optim=Optim(torch.tensor(1),warmup_step=4000,d_model=512)
    lrs=[]
    for i in range(40000):
        lrs.append(optim.get_lr())
    plt.plot(lrs)
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()