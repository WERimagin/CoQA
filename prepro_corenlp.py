#SQuADのデータ処理
#必要条件:CoreNLP
#Tools/core...で
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

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

from func.tf_idf import tf_idf
from func.corenlp import CoreNLP

def head_find(tgt):
    q_head=["what","how","who","when","which","where","why","whose","whom","is","are","was","were","do","did","does"]
    tgt_tokens=word_tokenize(tgt)
    true_head="<none>"
    for h in q_head:
        if h in tgt_tokens:
            true_head=h
            break
    return true_head

def modify(sentence,question_interro):
    #head=head_find(question)
    """
    if answer in sentence:
        sentence=sentence.replace(answer," ans_rep_tag ")
    """
    #sentence=" ".join([sentence,"ans_pos_tag",answer,"interro_tag",question_interro])
    sentence=" ".join([sentence,"interro_tag",question_interro])
    return sentence

def modify_history(history,now):
    #head=head_find(question)
    """
    if answer in sentence:
        sentence=sentence.replace(answer," ans_rep_tag ")
    """
    #sentence=" ".join([sentence,"ans_pos_tag",answer,"interro_tag",question_interro])
    sentence=" ".join([history,"history_append_tag",now])
    return sentence

def history_maker(neg_interro,question_interro):
    interro_list=["what","where","who","why","which","whom","how",""]
    while True:
        index=random.randrange(len(interro_list))
        if interro_list[index]!=question_interro.split()[0]:
            break
    question=interro_list[index]+" "+neg_interro
    return question


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
    sent_start_id=-1
    sent_end_id=-1
    start_id_list=[context_text.find(sent) for sent in context]
    end_id_list=[start_id_list[i+1] if i+1!=len(context) else len(context_text) for i,sent in enumerate(context)]
    for i,sent in enumerate(context):
        start_id=start_id_list[i]
        end_id=end_id_list[i]
        if start_id<=answer_start and answer_start<=end_id:
            sent_start_id=i
        if start_id<=answer_end and answer_end<=end_id:
            sent_end_id=i


    print(sent_start_id,sent_end_id)


    if sent_start_id==-1 or sent_end_id==-1:
        #sys.exit(-1)
        print("error")
        #sys.exit(-1)
    answer_sent=" ".join(context[sent_start_id:sent_end_id+1])
    #ここで答えを置換する方法。ピリオドが消滅した場合などに危険なので止める。
    """
    if answer_replace:
        context_text=context_text.replace(context[answer_start:answer_end],"<answer_word>")
        answer_sent=sent_tokenize(context_text)[sent_start_id]
    """
    return answer_sent


def data_process(input_path,out_path):
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

    dump_data=[]
    corenlp=CoreNLP()

    for paragraph in tqdm(data["data"]):
        context_text=paragraph["story"].lower()
        question_history=[]
        interro_history=[]
        for i in range(len(paragraph["questions"])):
            question_dict=paragraph["questions"][i]
            answer_dict=paragraph["answers"][i]
            question_text=question_dict["input_text"].lower()
            answer_text=answer_dict["input_text"].lower()
            question_history.append(question_text)

            span_start=answer_dict["span_start"]
            span_end=answer_dict["span_end"]
            span_text=answer_dict["span_text"]
            turn_id=paragraph["questions"][i]["turn_id"]

            question_interro,neg_interro,vb_check=corenlp.forward_verbcheck(question_text)#疑問詞を探してくる
            #if vb_check==False:
            #    print(question_history[-2:])


            if question_interro!="none_tag" and question_interro.rstrip()[-1]!="?":
                question_interro=question_interro+" ?"
            dump_data.append({"question_interro":question_interro,
                                "neg_interro":neg_interro,
                                "vb_check":vb_check})





    print(len(dump_data))




    with open(out_path,"w")as f:
        json.dump(dump_data,f)

#main
version="1.1"
type=""
question_modify=True
question_interro=True


data_process(input_path="data/coqa-train-v1.0.json",
            out_path="data/coqa-src-train-corenlp.txt"
            )
"""
data_process(input_path="data/coqa-dev-v1.0.json",
            out_path="data/coqa-src-dev-corenlp.txt"
            )
"""
