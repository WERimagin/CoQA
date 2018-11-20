#Bidafのtrain用のコード、testデータも同時に投げて評価してもいいかも。
#現在、epoch=9で0.89それ以下はほぼ収束

#流れ
#0.データの処理->prepro.shで実行、dataから必要なデータを取り出しpickle化、word2id,id2vecの処理
#1.contexts,questionsを取り出しid化
#2.dataloaderからbatchを取り出し(ただのshuffleされたid列)、それに従いbatchを作成してtorch化
#3.モデルに入れてp1,p2(スタート位置、エンド位置を出力)
#4.predictはp1,p2それぞれのargmaxを取り、それと正解の位置を比較して出力する

import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append("../")
from tqdm import tqdm
import nltk
import pickle
import json

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import time
from question_maker.model.seq2seq import Seq2Seq
from func.utils import Word2Id,DataLoader,make_vec,make_vec_c,to_var
import nltk


def data_loader(args,path,first=True):
    with open(path,"r")as f:
        t=json.load(f)
        contexts=t["contexts"]
        questions=t["questions"]
        answers=t["answers"]
        sentences=t["sentences"]
    with open("data/word2id2vec.json","r")as f:
        t=json.load(f)#numpy(vocab_size*embed_size)
        word2id=t["word2id"]
        id2vec=t["id2vec"]
        char2id=t["char2id"]

    #data_size=len(contexts)
    if first:
        data_size=len(questions)
    else:
        data_size=len(questions)

    id2vec=np.array(id2vec)

    print(len(answers))

    contexts_id=[[word2id[w] if w in word2id else 1 for w in sent] for sent in contexts[0:data_size]]
    questions_id=[[word2id[w] if w in word2id else 1 for w in sent] for sent in questions[0:data_size]]
    answers_id=[[word2id[w] if w in word2id else 1 for w in sent] for sent in answers[0:data_size]]
    sentences_id=[[word2id[w] if w in word2id else 1 for w in sent] for sent in sentences[0:data_size]]

    #id2word={i:w for w,i in word2id.items()}



    data={"contexts_id":contexts_id,
        "questions_id":questions_id,
        "answers_id":answers_id,
        "sentences_id":sentences_id}

    if first:
        args.c_vocab_size=len(char2id)
        args.pretrained_weight=id2vec
        args.vocab_size=id2vec.shape[0]
        args.embed_size=id2vec.shape[1]

    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_epoch", type=int, default="0", help="input model epoch")
    args = parser.parse_args()
    args.epoch_num=20
    args.train_batch_size=16
    args.test_batch_size=16
    args.hidden_size=100
    args.c_embed_size=20
    args.dropout=0.2
    args.self_attention=True
    args.lr=0.001
    args.gpu_num=3

    return args

def loss_calc(predict,target):
    criterion = nn.CrossEntropyLoss(ignore_index=0)#<pad>=0を無視
    batch=predict.size(0)
    seq_len=predict.size(1)
    predict=predict.contiguous().view(batch*seq_len,-1)
    target=target.contiguous().view(-1)
    loss=criterion(predict,target)
    return loss

def predict_calc(predict,target):
    #predict:(batch,seq_len,embed_size)
    #target:(batch,seq_len)
    type="bleu"
    if type=="normal":
        batch=predict.size(0)
        seq_len=predict.size(1)
        predict=predict.contiguous().view(batch*seq_len,-1)
        target=target.contiguous().view(-1)
        predict_rate=(torch.argmax(predict,dim=-1)==target).sum().item()
        return predict_rate
    elif type=="bleu":
        predict=torch.argmax(predict,dim=-1).tolist()#(batch,seq_len,embed_size)
        target=target.tolist()#(batch,seq_len)
        predict_sum=0
        for p,t in zip(predict,target):#batchごと
            predict_sum+=nltk.bleu_score.sentence_bleu([p],t)
        return predict_sum


def model_handler(args,data,train=True):
    start=time.time()
    questions_id=data["questions_id"]
    sentences_id=data["sentences_id"]
    contexts_id=data["contexts_id"]

    data_size=len(questions_id)
    if train:
        batch_size=args.train_batch_size
        model.train()
    else:
        batch_size=args.test_batch_size
        model.eval()
    dataloader=DataLoader(data_size,batch_size,train)
    batches=dataloader()
    predict_rate=0
    for i_batch,batch in tqdm(enumerate(batches)):
        """
        if i_batch==10:
            batch[0]=87060
        elif i_batch==20:
            batch[0]=87063
        """
        print(batch)
        #batch:(context,question,answer_start,answer_end)*N
        #これからそれぞれを取り出し処理してモデルへ
        #c_words=make_vec([sentences_id[i] for i in batch])#(batch,seq_len)
        c_words=make_vec([contexts_id[i] for i in batch])#(batch,seq_len)
        q_words=make_vec([questions_id[i] for i in batch])
        with open("log.txt","a")as f:
            f.write(" ".join([str(w) for w in batch]))
            f.write("c:{}\tq:{}\n".format(c_words.size(),q_words.size()))
        if train:
            optimizer.zero_grad()
        predict=model(c_words,q_words,train=True)#(batch,seq_len,vocab_size)
        if train:
            loss=loss_calc(predict,q_words)#batch*seq_lenをして内部で計算
            loss.backward()
            optimizer.step()
            if i_batch%10==0:
                with open("log.txt","a")as f:
                    now=time.time()
                    f.write("epoch,{}\tbatch\t{}\tloss:{}\ttime:{}\n".format(epoch,i_batch,loss.data,now-start))
        else:
            predict_rate+=predict_calc(predict,q_words)

    with open("log.txt","a")as f:
        if train:
            f.write("epoch:{}\ttime:{}\n".format(epoch,time.time()-start))
            torch.save(model.state_dict(), 'model_data/epoch_{}_model.pth'.format(epoch))
        else:
            f.write("predict_rate:{}\n".format(predict_rate/data_size))


##start main

args=get_args()

train_data=data_loader(args,"data/squad_train_data.json",first=True)
test_data=data_loader(args,"data/squad_test_data.json",first=False)


model=Seq2Seq(args)

if args.start_epoch>=1:
    param = torch.load("model_data/epoch_{}_model.pth".format(args.start_epoch-1))
    model.load_state_dict(param)
else:
    args.start_epoch=0

"""
if torch.cuda.is_available():
    model.cuda(3)
else:
    print("cant use cuda")
"""
#pytorch0.4より、OpenNMT参考
device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(),lr=args.lr)

for epoch in range(args.start_epoch,args.epoch_num):
    print("start")
    model_handler(args,train_data,True)
    model_handler(args,train_data,False)
    model_handler(args,test_data,False)
