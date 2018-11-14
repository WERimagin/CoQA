#testデータによる評価
#epoch数を0~変化させて評価。modelはmodel/に置いてある物を使う。

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
parser.add_argument("--model", type=str, default="", help="input model name")
args = parser.parse_args()

with open("data/test_data.pickle","rb")as f:
    t=pickle.load(f)
    contexts=t[0]
    questions=t[1]
    answer_starts=t[2]
    answer_ends=t[3]
    answer_texts=t[4]
    ids=t[5]
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
args.batch_size=4
args.hidden_size=100#=100
args.embed_size=id2vec.shape[1]#=300
args.c_embed_size=100
args.vocab_size=id2vec.shape[0]
args.c_vocab_size=len(char2id)
args.pretrained_weight=id2vec
args.dropout=0.2


model=Self_Attention_Bidaf(args)

with open("test_log.txt","a")as f:
    f.write("data_size={}\tbatch_size={}\n".format(data_size,args.batch_size))
    f.write("hidden_size={}\tembed_size={}\n".format(args.hidden_size,args.embed_size))
    f.write("start Test\n")

for epoch in range(10):
    start=time.time()
    if torch.cuda.is_available():
        print("use cuda")
        param = torch.load("model/epoch_{}_model.pth".format(epoch))
        model.load_state_dict(param)
        model.cuda()
    else:
        print("no use cuda")
        #param = torch.load(args.model,map_location='cpu')
        model.load_state_dict(param)

    dataloader=DataLoader(data_size,args.batch_size,shuffle=False)
    model.eval()

    p1_predict=0
    p2_predict=0
    answer_dict={}
    batches=dataloader()
    for batch in tqdm(batches):
        c_words=make_vec([contexts_id[i] for i in batch])
        q_words=make_vec([questions_id[i] for i in batch])
        c_chars=make_vec_c([contexts_c_id[i] for i in batch])
        q_chars=make_vec_c([questions_c_id[i] for i in batch])
        a_starts=to_var(torch.from_numpy(np.array([answer_starts[i] for i in batch],dtype="long")))
        a_ends=to_var(torch.from_numpy(np.array([answer_ends[i] for i in batch],dtype="long")))
        id_list=([ids[i] for i in batch])
        p1,p2=model(c_words,q_words,c_chars,q_chars)
        p1_predict+=(torch.argmax(p1,dim=-1)==a_starts).sum().item()
        p2_predict+=(torch.argmax(p2,dim=-1)==a_ends).sum().item()

    with open("test_log.txt","a")as f:
        f.write("epoch:{}\tp1_predict{}\tp2_predict:{}\ttime:{}\n".format(epoch,p1_predict/data_size,p2_predict/data_size,time.time()-start))
