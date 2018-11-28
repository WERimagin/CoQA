#SQuADのデータ処理

import os
import sys
sys.path.append("../")
import json
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize,sent_tokenize
import pickle
import collections



def c2wpointer(context_text,context,answer_start,answer_end):#answer_start,endをchara単位からword単位へ変換
    #nltk.tokenizeを使って分割
    #ダブルクオテーションがなぜか変化するので処理
    token_id={}
    cur_id=0
    for i,token in enumerate(context):
        start=context_text.find(token,cur_id)
        token_id[i]=(start,start+len(token))
        cur_id=start+len(token)
    for i in range(len(token_id)):
        if token_id[i][0]<=answer_start and answer_start<=token_id[i][1]:
            answer_start_w=i
            break
    for i in range(len(token_id)):
        if token_id[i][0]<=answer_end and answer_end<=token_id[i][1]:
            answer_end_w=i
            break
    return answer_start_w,answer_end_w

#sentenceを受け取り、tokenizeして返す
def tokenize(sent):
    return [token.replace('``','"').replace("''",'"') for token in word_tokenize(sent)]

#context_textを文分割して、answer_start~answer_end(char単位)のスパンが含まれる文を返す
#やってることはc2iと多分同じアルゴリズう
def answer_find(context_text,answer_start,answer_end):
    context=sent_tokenize(context_text)
    current_p=0
    for i,sent in enumerate(context):
        end_p=current_p+len(sent)
        if current_p<=answer_start and answer_start<=end_p:
            sent_start_id=i
        if current_p<=answer_end and answer_end<=end_p:
            sent_end_id=i
        current_p+=len(sent)+1#スペースが消えている分の追加、end_pの計算のところでするべきかは不明

    answer_sent=word_tokenize(" ".join(context[sent_start_id:sent_end_id+1]))
    return answer_sent


def data_process(input_path,src_path,tgt_path,word_count,lower=True):
    with open(input_path,"r") as f:
        data=json.load(f)
    contexts=[]
    questions=[]
    answer_starts=[]
    answer_ends=[]
    answer_texts=[]
    answers=[]
    sentences=[]
    ids=[]
    word2count=collections.Counter()
    char2count=collections.Counter()

    for topic in tqdm(data["data"]):
        topic=topic["paragraphs"]
        for paragraph in topic:
            context_text=paragraph["context"]
            context=tokenize(context_text)
            context.append("<eos>")
            #メモリサイズの確保のため、サイズが大きいcontextはスキップ
            #結果的に、87599個のcontextのうち、500をカット
            if word_count:
                for word in context:
                    word2count[word]+=1
                    for char in word:
                        char2count[char]+=1
            for qas in paragraph["qas"]:
                question_text=qas["question"]
                question_text=" ".join(word_tokenize(question_text))
                if len(qas["answers"])==0:
                    continue
                a=qas["answers"][0]
                answer_start=a["answer_start"]
                answer_end=a["answer_start"]+len(a["text"])
                answer_sent=answer_find(context_text,answer_start,answer_end)#contextの中からanswerが含まれる文を見つけ出す
                answer_sent=" ".join(answer_sent)
                questions.append(question_text)
                sentences.append(answer_sent)

    print(len(questions),len(sentences))

    with open(src_path,"w")as f:
        sentences="\n".join(sentences)
        f.write(sentences)

    with open(tgt_path,"w")as f:
        questions="\n".join(questions)
        f.write(questions)

def vec_process(contexts,word2id,char2id):

    vec_size=100

    #vec==300は単語が空白で区切られているものがあり、要対処
    if vec_size==300:
            path="data/glove.840B.300d.txt"
    else:
        path="data/glove.6B.{}d.txt".format(vec_size)

    w2vec={}
    id2vec=np.zeros((len(list(word2id.items())),vec_size))

    if os.path.exists(path)==True:
        with open(path,"r")as f:
            for line in tqdm(f):
                line_split=line.split()
                w2vec[line_split[0]]=[float(i) if i!="." else float(0)  for i in line_split[1:]]

        for w,i in word2id.items():
            if w.lower() in w2vec:
                id2vec[i]=w2vec[w.lower()]



    with open("data/word2id2vec.json","w")as f:
        t={"word2id":word2id,
            "id2vec":id2vec.tolist(),
            "char2id":char2id}
        json.dump(t,f)

#main
version="1.1"
data_process(input_path="data/squad_train-v{}.json".format(version),src_path="data/squad-src-train.txt",tgt_path="data/squad-tgt-train.txt",word_count=True,lower=True)
data_process(input_path="data/squad_dev-v{}.json".format(version),src_path="data/squad-src-dev.txt",tgt_path="data/squad-tgt-dev.txt",word_count=True,lower=True)

#python preprocess.py -train_src data/squad-src-train.txt -train_tgt data/squad-tgt-train.txt -valid_src data/squad-src-val.txt -valid_tgt data/squad-tgt-val.txt -save_data data/demo
