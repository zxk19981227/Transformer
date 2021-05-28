import heapq
import copy
def beam_search(src,beam_size,device,model,trg):
    batch_size=src.shape[1]
    # input_mask=(src==0)
    memory,mask=model.encode(src)#memory.shape:seq_len,batch_size,embedding_size
    # mask=(src==0).to(device)
    # memory,mask=model.Encode(src,device)
    total_predict=[[[2]]]*batch_size
    total_score=[[0]]*batch_size
    total_length=[[0]]*batch_size
    total_is_end=[[False]]*batch_size
    if beam_size==1:
        beam_memory=memory
        beam_mask=mask
    else:
        beam_memory=[]
        beam_mask=[]
        tmp=copy.deepcopy(memory)
        for i in range(tmp.shape[1]):
            for j in range(beam_size):
                beam_memory.append(tmp[:,i])
                beam_mask.append(mask[i])
        beam_memory=torch.stack(beam_memory,1)
        # print(memory.shape)
        beam_mask=torch.stack(beam_mask,0)
    for i in range(100):
        if i ==0:
            current_memory=memory
            current_mask=mask
        else:
            current_memory=beam_memory
            current_mask=beam_mask
        current_predict=total_predict
        current_score=total_score
        current_is_end=total_is_end
        current_length=total_length
        total_is_end=[]
        total_length=[]
        total_predict=[]
        total_score=[]
        if i == 0:
            current_num = 1
        else:
            current_num = beam_size
        current_input=torch.tensor(current_predict).view(-1,i+1).to(device).t()
        # print(current_input.shape)
        # print(current_memory.shape)
        # print(current_mask.shape)
        out=model.linear(model.decode(current_memory,current_input,current_mask))*model.power_value
        out=out[-1]
        #out.shape :batch_size,target
        out=torch.log_softmax(out,-1)
        scores,indexs=torch.topk(out,beam_size,-1)
        scores=scores.cpu().numpy().tolist()
        indexs=indexs.cpu().numpy().tolist()
        # if 1 in indexs:
        #     scores, indexs = torch.topk(out, beam_size+1, -1)
        #     indexs=indexs.cpu().numpy().tolist()
        #     scores=scores.cpu().numpy().tolist()
        #     current_p=indexs.index(1)
        #     indexs.pop(current_p)
        #     scores.pop(current_p)

        #score,indexs:shape(batch*current_num,beam_size)
        next_is_end = []
        next_length = []
        next_score = []
        next_predict = []
        for j in range(out.shape[0]):
            sentence_num=j//current_num
            score_num=j%current_num
            if score_num==0:
                next_is_end=[]
                next_length=[]
                next_score=[]
                next_predict=[]
            if current_is_end[sentence_num][score_num]:#如果句子已经结束了
                this_score=current_score[sentence_num][score_num]
                this_length=current_length[sentence_num][score_num]
                this_is_end=True
                this_predict=current_predict[sentence_num][score_num]+[2]
                next_score.append(this_score)
                next_length.append(this_length)
                next_predict.append(this_predict)
                next_is_end.append(this_is_end)
            else:
                for score,index in zip(scores[j],indexs[j]):
                    this_score=((current_score[sentence_num][score_num])*math.pow(current_length[sentence_num][score_num],0.7)+score)/math.pow(current_length[sentence_num][score_num]+1,0.7)
                    this_length=current_length[sentence_num][score_num]+1
                    this_predict=current_predict[sentence_num][score_num]+[index]
                    if index==1:
                        this_is_end=True
                    else:
                        this_is_end=False
                    next_score.append(this_score)
                    next_length.append(this_length)
                    next_predict .append(this_predict)
                    next_is_end.append(this_is_end)
            if (score_num+1)%current_num==0:

                    #已经计算了beam_size*beam_size个信息了
                max_score_index=heapq.nlargest(beam_size,range(len(next_score)),next_score.__getitem__)
                tmp_is_end=[next_is_end[i] for i in max_score_index]
                tmp_length=[next_length[i] for i in max_score_index]
                tmp_score=[next_score[i] for i in max_score_index]
                tmp_predict=[next_predict[i] for i in max_score_index]
                total_score.append(tmp_score)
                total_is_end.append(tmp_is_end)
                total_length.append(tmp_length)
                total_predict.append(tmp_predict)
        flag=False
        for i in range(len(total_is_end)):
            for j in range(len(total_is_end[i])):
                if not total_is_end[i][j]:
                    flag=True
                    break
            if flag:
                break
        if not flag:
            break
        else:
            continue
    result=[]
    for sens,scores in zip(total_predict,total_score):
        ind=scores.index(max(scores))
        result.append(sens[ind])
    return result
import argparse
import math
import dill as pickle
# from base_beam_search import beamsearch
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from model2 import Model

# from Transformer_base_new import NMT as transformer
# from Transformer import Transformer
import torch
import torch.nn.functional as F
# from torchtext.data import Field, Dataset, BucketIterator
# from Componments.optim import  ScheduledOptim
# fitlog.set_log_dir('./logs')
from torch.utils.data import DataLoader
from data_load import Data
def eval(eval_path,device):
    with open("./data/en.txt") as f:
        lines=f.readlines()
    en_dict={}
    for line in lines:
        en_dict[len(en_dict.keys())]=line.strip()
    with open("./data/de.txt") as f:
        lines=f.readlines()
    de_dict={}
    for line in lines:
        de_dict[len(de_dict.keys())]=line.strip()
    paths=eval_path.split('/')
    file_path='./translate/'+paths[-1]
    in_file=open(file_path+'_ref.txt','w')
    out_file=open(file_path+"_tar.txt",'w')
    def collate_fn(batch):
        src=[]
        trg=[]
        for s,t in batch:#(batch_size,seq_len)
            src.append(s)
            trg.append(t.numpy().tolist())
            # trg.append(t)
        src=torch.nn.utils.rnn.pad_sequence(src,batch_first=False,padding_value=2)
        # trg=torch.nn.utils.rnn.pad_sequence(trg,batch_first=True,padding_value=2)
        return src,trg
    path="./data/"
    src_vocab_size = len(open(path + "de.txt").readlines())
    trg_vocab_size = len(open(path + "en.txt").readlines())
    model = Model(input_vocab_size=src_vocab_size, output_vocab_size=trg_vocab_size,device=device)
    # val_iter=data("./final/dev.txt")
    val_iter=DataLoader(Data("./data/only_de_sen_test.txt","./data/only_en_sen_test.txt"),batch_size=49,shuffle=False,collate_fn=collate_fn)
    # val_iter=DataLoader(Data("./data/test_de.txt","./data/test_en.txt"),batch_size=49,shuffle=False,collate_fn=collate_fn)
    model=model.to(device)
    # model.load_state_dict(torch.load("./pmodel_final_version.bin"))
    model.load_state_dict(torch.load(eval_path))

    batch_size=49
    # val_iter=DataLoader(val_iter,batch_size,False,collate_fn=collate_fn)
    # val_iter=DataLoader(Data2(["./data/de_test.txt","./data/en_test.txt"]),batch_size=1,shuffle=True,collate_fn=collate_fn)

    beam_size=4
    max_len=50
    total=[]
    pre=[]
    with torch.no_grad():
        model.eval()
        with tqdm(total=len(val_iter)) as t:
            for src,trg in val_iter:
                src=src.to(device)
                # print()
                # predict=model(src,trg[:,:1])
                # bleu=beamsearch(model,src,batch_size,4,100,device,trg_vocab_size)

                sentence=beam_search(src,beam_size,device,model,trg)
                for i in range(len(sentence)):
                    while 2 in sentence[i]:
                        sentence[i].remove(2)
                # trg=trg.unsqueeze(1)
                # trg=trg.numpy().tolist()
                for i in range(len(trg)):
                    while 2 in trg[i]:
                        trg[i].remove(2)
                trg=[[each] for each in trg]
                trg2=[[en_dict[i] for i in each[0]] for each in trg]
                trg2=[' '.join(each) for each in trg2]
                trg2='\n'.join(trg2)+'\n'
                in_file.write(trg2)

                out_sentence=[[en_dict[i] for i in each] for each in sentence]
                out_sentence=[' '.join(each) for each in out_sentence]
                out_sentence='\n'.join(out_sentence)+'\n'
                out_file.write(out_sentence)
                # print(len(trg))
                # print(len(sentence))
                # print(trg)
                # print(sentence)
                bleu=corpus_bleu(trg,sentence)
                # print(trg)
                # print(sentence)
                total=total+trg
                # print(pre)
                # print(sentence)
                pre=pre+sentence
                t.set_postfix(bleu=bleu)
                t.update(1)

            # bleu=beamsearch(model,val_iter,8,beam_size,max_len,device)
        # fitlog.add_best_metric(bleu,"bleu")
    in_file.close()
    out_file.close()
    bleu=corpus_bleu(total,pre)
    print("best belu is{}".format(bleu))
    return bleu