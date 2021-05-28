from torch.optim import Adam
from numpy import mean
from Model import Transformer
from utils import Optim, get_args, cal_loss, cal_accruacy, collate_fn
from torch.utils.data import DataLoader
import torch
from dataload import Dataload
import pickle
from tqdm import tqdm
import math
import fitlog
import numpy as np
import random
fitlog.set_log_dir('./logs')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1)
def train_step(src, trg, model, optim, device):
    optim.zero_grad()
    src = src.to(device)
    trg = trg.to(device)
    trg_input = trg[:, :-1]
    trg_label = trg[:, 1:]
    predict = model(src, trg_input)
    loss = cal_loss(predict, trg_label)
    loss.backward()
    optim.step()
    correct, total = cal_accruacy(predict, trg_label)
    return loss.item(), total, correct


def train(step, model, data_loader, optim, device):
    model.train()
    total = 0

    correct = 0
    losses = []
    pbtr = tqdm(total=len(data_loader))
    for src, trg in data_loader:
        loss, to, cor = train_step(src, trg, model, optim, device)
        total += to
        correct += cor
        losses.append(loss)
        pbtr.update(1)
        pbtr.set_postfix({'accuracy': cor / to, "loss": loss, "ppl": math.exp(loss)})
    pbtr.close()
    print("training epoch {} ||  ppl {} ||accuracy {} || loss  {}".format(step, math.exp(mean(losses)), (correct / total),
                                                                          mean(losses)))
    fitlog.add_loss(math.exp(mean(losses)), step=step, name='train')
    fitlog.add_metric(correct / total, step=step, name='train accuracy')


def eval_step(src, trg, model, device):
    src = src.to(device)
    trg = trg.to(device)
    trg_input = trg[:, :-1]
    trg_label = trg[:, 1:]
    predict = model(src, trg_input)
    loss = cal_loss(predict, trg_label)
    correct, total = cal_accruacy(predict, trg_label)
    return loss.item(), total, correct


def eval(step, model, data_loader, best_loss, device):
    model.eval()
    total = 0
    correct = 0
    losses = []
    pbtr = tqdm(total=len(data_loader))
    for src, trg in data_loader:
        loss, to, cor = eval_step(src, trg, model, device)
        total += to
        correct += cor
        losses.append(loss)
        pbtr.update(1)
        # pbtr.set_postfix({'accuracy':cor/to,"loss":loss,"ppl":math.exp(loss)})
    pbtr.close()
    fitlog.add_loss(math.exp(mean(losses)), step=step, name='eval')
    fitlog.add_metric(correct / total, step=step, name='eval accuracy')
    if best_loss > mean(losses):
        torch.save(model.state_dict(), 'best_model.pkl')
        best_loss = mean(losses)
        print("saving to best_model.pkl")
    print("eval epoch {} ||  ppl {} ||accuracy {} || loss  {} ||best loss {}".format(step, math.exp(mean(losses)),
                                                                                     correct / total, mean(losses),
                                                                                     best_loss))
    return best_loss


def main():
    args = get_args()
    src_vocab_size = len(pickle.load(open(args.data_bin + '/dict' + "." + args.src_lang, 'rb')).keys())
    tgt_vocab_size = len(pickle.load(open(args.data_bin + '/dict' + '.' + args.tgt_lang, 'rb')).keys())
    device = 'cuda'
    model = Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
                        encoder_layer_num=args.encoder_layer_num, decoder_layer_num=args.decoder_layer_num,
                        hidden_size=args.hidden_size, feedback_size=args.feedback, num_head=args.num_head,
                        dropout=args.dropout,device=device)
    optim = Optim(Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9), warmup_step=4000, d_model=args.hidden_size)
    train_loader = DataLoader(Dataload(args.data_bin + '/' + 'train', args.src_lang, args.tgt_lang),
                              batch_size=args.batch_size, collate_fn=collate_fn,shuffle=True)
    # optim = Adam(model.parameters(), lr=5e-6)
    test_loader = DataLoader(Dataload(args.data_bin + '/' + 'test', args.src_lang, args.tgt_lang),
                             batch_size=args.batch_size, collate_fn=collate_fn)
    valid_loader = DataLoader(Dataload(args.data_bin + '/' + 'valid', args.src_lang, args.tgt_lang),
                              batch_size=args.batch_size, collate_fn=collate_fn)
    best_loss = 1e4
    model = model.to(device)
    model.load_state_dict(torch.load('/home/hejun/zxk/transformer/source/best_model.pkl'))
    for i in range(args.epoch):
        # train(i, model, data_loader=train_loader, optim=optim, device=device)
        with torch.no_grad():
            best_loss = eval(i, model, valid_loader, best_loss, device)
    # torch.save(model.state_dict(), 'best_model.pkl')
main()
