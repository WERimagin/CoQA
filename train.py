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


parser = argparse.ArgumentParser()
parser.add_argument("--start_epoch", type=int, default="0", help="input model epoch")
args = parser.parse_args()


with open("data/train_data.pickle","rb")as f:
    t=pickle.load(f)
    contexts=t[0]
    questions=t[1]
    answer_starts=t[2]
    answer_ends=t[3]
with open("data/word2id2vec.pickle","rb")as f:
    t=pickle.load(f)#numpy(vocab_size*embed_size)
    word2id=t[0]
    id2vec=t[1]
    char2id=t[2]

data_size=len(contexts)

contexts_id=[[word2id[w] if w in word2id else 1  for w in sent] for sent in contexts[0:data_size]]
questions_id=[[word2id[w] if w in word2id else 1  for w in sent] for sent in questions[0:data_size]]
contexts_c_id=[[[char2id[c] if c in char2id else 1 for c in w] for w in sent] for sent in contexts[0:data_size]]
questions_c_id=[[[char2id[c] if c in char2id else 1 for c in w] for w in sent] for sent in questions[0:data_size]]
answer_starts=answer_starts[0:data_size]
answer_ends=answer_ends[0:data_size]


args.epoch_num=12
args.batch_size=32
args.hidden_size=100#=100
args.embed_size=id2vec.shape[1]#=300
args.c_embed_size=20
args.vocab_size=id2vec.shape[0]
args.c_vocab_size=len(char2id)
args.pretrained_weight=id2vec
args.dropout=0.2

#model周りの変数やoptimizer定義など
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

optimizer = optim.Adam(model.parameters(),lr=0.0005)
criterion = nn.CrossEntropyLoss()

model.train()
dataloader=DataLoader(data_size,args.batch_size)
start=time.time()

with open("log.txt","a")as f:
    f.write("data_size={}\tbatch_size={}\n".format(data_size,args.batch_size))
    f.write("hidden_size={}\tembed_size={}\n".format(args.hidden_size,args.embed_size))
    f.write("start Training\n")

for epoch in range(args.start_epoch,args.epoch_num):
    p1_predict=0
    p2_predict=0
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
        print(c_words.size())
        optimizer.zero_grad()
        p1,p2=model(c_words,q_words,c_chars,q_chars,train=True)
        p1_loss=criterion(p1,a_starts)
        p2_loss=criterion(p2,a_ends)
        (p1_loss+p2_loss).backward()
        optimizer.step()
        p1_predict+=(torch.argmax(p1,dim=-1)==a_starts).sum().item()
        p2_predict+=(torch.argmax(p2,dim=-1)==a_ends).sum().item()
        if i_batch%500==0:
            with open("log.txt","a")as f:
                now=time.time()
                f.write("epoch,{}\tbatch\t{}\ttime:{}\n".format(epoch,i_batch,now-start))
    if epoch%1==0:
        now=time.time()
        with open("log.txt","a")as f:
            f.write("epoch:{}\ttime:{}\n".format(epoch,now-start))
            f.write("p1_predict{}\tp2_predict:{}\n".format(p1_predict/data_size,p2_predict/data_size))

        start=now
        torch.save(model.state_dict(), 'model/epoch_{}_model.pth'.format(epoch))
