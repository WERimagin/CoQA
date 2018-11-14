#Bidafのtrain用のコード、testデータも同時に投げて評価してもいいかも。
#現在、epoch=9で0.89それ以下はほぼ収束

#流れ
#0.データの処理->prepro.shで実行、dataから必要なデータを取り出しpickle化、word2id,id2vecの処理
#1.contexts,questionsを取り出しid化
#2.dataloaderからbatchを取り出し(ただのshuffleされたid列)、それに従いbatchを作成してtorch化
#3.モデルに入れてp1,p2(スタート位置、エンド位置を出力)
#4.predictはp1,p2それぞれのargmaxを取り、それと正解の位置を比較して出力する

from tqdm import tqdm
import nltk
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import time
from func.layer import Bidaf
from func.self_attention_layer import Self_Attention_Bidaf
from func.utils import Word2Id,DataLoader,make_vec,make_vec_c,to_var
from func.ema import EMA


def data_loader(args,path,first=True):
    with open(path,"rb")as f:
        t=pickle.load(f)
        contexts=t[0]
        questions=t[1]
        answer_starts=t[2]
        answer_ends=t[3]
        answer_texts=t[4]
    with open("data/word2id2vec.pickle","rb")as f:
        t=pickle.load(f)#numpy(vocab_size*embed_size)
        word2id=t[0]
        id2vec=t[1]
        char2id=t[2]

    #data_size=len(contexts)
    if first:
        data_size=len(contexts)
    else:
        data_size=len(contexts)

    contexts_id=[[word2id[w] if w in word2id else 1  for w in sent] for sent in contexts[0:data_size]]
    questions_id=[[word2id[w] if w in word2id else 1  for w in sent] for sent in questions[0:data_size]]
    contexts_c_id=[[[char2id[c] if c in char2id else 1 for c in w] for w in sent] for sent in contexts[0:data_size]]
    questions_c_id=[[[char2id[c] if c in char2id else 1 for c in w] for w in sent] for sent in questions[0:data_size]]
    answer_starts=answer_starts[0:data_size]
    answer_ends=answer_ends[0:data_size]
    answer_texts=[[word2id[w] if w in word2id else 1  for w in sent] for sent in answer_texts[0:data_size]]

    data={"contexts_id":contexts_id,
        "questions_id":questions_id,
        "contexts_c_id":contexts_c_id,
        "questions_c_id":questions_c_id,
        "answer_starts":answer_starts,
        "answer_ends":answer_ends}

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
    args.epoch_num=12
    args.train_batch_size=16
    args.test_batch_size=4
    args.hidden_size=100
    args.c_embed_size=20
    args.dropout=0.2
    args.self_attention=True
    args.lr=0.0005

    return args

def model_handler(args,data,train=True):
    start=time.time()
    contexts_id=data["contexts_id"]
    questions_id=data["questions_id"]
    contexts_c_id=data["contexts_c_id"]
    questions_c_id=data["questions_c_id"]
    answer_starts=data["answer_starts"]
    answer_ends=data["answer_ends"]
    context_answer_id=
    p1_predict=0
    p2_predict=0
    data_size=len(contexts_id)
    if train:
        batch_size=args.train_batch_size
        model.train()
    else:
        batch_size=args.test_batch_size
        model.eval()
    dataloader=DataLoader(data_size,batch_size,train)
    batches=dataloader()
    for i_batch,batch in tqdm(enumerate(batches)):
        #batch:(context,question,answer_start,answer_end)*N
        #これからそれぞれを取り出し処理してモデルへ
        c_words=make_vec([contexts_id[i] for i in batch])
        q_words=make_vec([questions_id[i] for i in batch])
        c_chars=make_vec_c([contexts_c_id[i] for i in batch])
        q_chars=make_vec_c([questions_c_id[i] for i in batch])
        a_starts=to_var(torch.from_numpy(np.array([answer_starts[i] for i in batch],dtype="long")))
        a_ends=to_var(torch.from_numpy(np.array([answer_ends[i] for i in batch],dtype="long")))
        print(q_chars.size())
        if train:
            optimizer.zero_grad()
        p1,p2=model(c_words,q_words,c_chars,q_chars,train=True)
        if train:
            p1_loss=criterion(p1,a_starts)
            p2_loss=criterion(p2,a_ends)
            (p1_loss+p2_loss).backward()
            optimizer.step()
            if i_batch%500==0:
                with open("log.txt","a")as f:
                    now=time.time()
                    f.write("epoch,{}\tbatch\t{}\ttime:{}\n".format(epoch,i_batch,now-start))
        else:
            p1_predict+=(torch.argmax(p1,dim=-1)==a_starts).sum().item()
            p2_predict+=(torch.argmax(p2,dim=-1)==a_ends).sum().item()

    with open("log.txt","a")as f:
        if train:
            f.write("epoch:{}\ttime:{}\n".format(epoch,time.time()-start))
            torch.save(model.state_dict(), 'model/epoch_{}_model.pth'.format(epoch))
        else:
            f.write("p1_predict{}\tp2_predict:{}\n".format(p1_predict/data_size,p2_predict/data_size))

##start main

args=get_args()

train_data=data_loader(args,"data/train_data.pickle",first=True)
test_data=data_loader(args,"data/test_data.pickle")

if args.self_attention:
    model=Self_Attention_Bidaf(args)
else:
    model=Bidaf(args)
if args.start_epoch>=1:
    param = torch.load("model/epoch_{}_model.pth".format(args.start_epoch-1))
    model.load_state_dict(param)
else:
    args.start_epoch=0
if torch.cuda.is_available():
    model.cuda()
else:
    print("cant use cuda")

optimizer = optim.Adam(model.parameters(),lr=args.lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(args.start_epoch,args.epoch_num):
    model_handler(args,train_data,True)
    model_handler(args,train_data,False)
    model_handler(args,test_data,False)
