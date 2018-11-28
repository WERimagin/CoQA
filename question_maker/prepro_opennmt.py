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

def head_find(tgt):
    q_head=["what","how","who","when","which","where","why","whose","whom","is","are","was","were","do","did","does"]
    tgt_tokens=word_tokenize(tgt)
    true_head="<none>"
    for h in q_head:
        if h in tgt_tokens:
            true_head=h
            break
    return true_head

def modify(sentence,question,answer,answer_replace):
    head=head_find(question)
    """
    if answer in sentence:
        sentence=sentence.replace(answer," ans_rep_tag ")
    """
    sentence=" ".join([sentence,"ans_pos_tag",answer,"inter_tag",head])
    return sentence


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
#やってることはc2iと多分同じアルゴリズム
def answer_find(context_text,answer_start,answer_end,answer_replace):
    context=sent_tokenize(context_text)
    current_p=0
    for i,sent in enumerate(context):
        end_p=current_p+len(sent)
        if current_p<=answer_start and answer_start<=end_p:
            sent_start_id=i
        if current_p<=answer_end and answer_end<=end_p:
            sent_end_id=i
        current_p+=len(sent)+1#スペースが消えている分の追加、end_pの計算のところでするべきかは不明
    answer_sent=" ".join(context[sent_start_id:sent_end_id+1])
    #ここで答えを置換する方法。ピリオドが消滅した場合などに危険なので止める。
    """
    if answer_replace:
        context_text=context_text.replace(context[answer_start:answer_end],"<answer_word>")
        answer_sent=sent_tokenize(context_text)[sent_start_id]
    """
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
    answer_replace=False
    word2count=collections.Counter()
    char2count=collections.Counter()
    for topic in tqdm(data["data"]):
        topic=topic["paragraphs"]
        for paragraph in topic:
            context_text=paragraph["context"].lower()
            for qas in paragraph["qas"]:
                question_text=qas["question"].lower()
                if len(qas["answers"])==0:
                    continue
                a=qas["answers"][0]
                answer=a["text"].lower()
                answer_start=a["answer_start"]
                answer_end=a["answer_start"]+len(a["text"])
                answer_sent=answer_find(context_text,answer_start,answer_end,answer_replace)#contextの中からanswerが含まれる文を見つけ出す
                """
                answer_sent=" ".join(tokenize(answer_sent))
                question_text=" ".join(tokenize(question_text))
                answer=" ".join(tokenize(answer))
                """
                answer_sent=modify(answer_sent,question_text,answer,answer_replace)#answwer_sentにanswerを繋げる。
                #tokenizeを掛けて処理
                answer_sent=" ".join(tokenize(answer_sent))
                question_text=" ".join(tokenize(question_text))
                answer=" ".join(tokenize(answer))
                questions.append(question_text)
                sentences.append(answer_sent)

    print(len(questions),len(sentences))

    with open(src_path,"w")as f:
        for s in sentences:
            f.write(s+"\n")

    with open(tgt_path,"w")as f:
        for s in questions:
            f.write(s+"\n")

#main
version="1.1"
type="sentence_answer"
data_process(input_path="data/squad_train-v{}.json".format(version),src_path="data/squad-src-train-{}.txt".format(type),tgt_path="data/squad-tgt-train-{}.txt".format(type),word_count=True,lower=True)
data_process(input_path="data/squad_dev-v{}.json".format(version),src_path="data/squad-src-dev-{}.txt".format(type),tgt_path="data/squad-tgt-dev-{}.txt".format(type),word_count=True,lower=True)

#python preprocess.py -train_src data/squad-src-train.txt -train_tgt data/squad-tgt-train.txt -valid_src data/squad-src-val.txt -valid_tgt data/squad-tgt-val.txt -save_data data/demo
