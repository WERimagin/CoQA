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


    #print(sent_start_id,sent_end_id)


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


def data_process(input_path,output_path,dict_path,modify_path):
    with open(input_path,"r") as f:
        data=json.load(f)
    with open(dict_path,"r") as f:
        corenlp_data=json.load(f)

    modify_data=[]
    with open(modify_path,"r") as f:
        for line in f:
            modify_data.append(line.rstrip())
    contexts=[]
    questions=[]
    answer_starts=[]
    answer_ends=[]
    answer_texts=[]
    answers=[]
    sentences=[]
    ids=[]
    answer_replace=False
    count=0
    modify_count=0
    for paragraph in tqdm(data["data"]):
        context_text=paragraph["story"].lower()
        question_history=[]
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

            d=corenlp_data[count]


            if d["vb_check"]==False and d["question_interro"]!="none_tag":
                h=corenlp_data[count-1]
                his_neg=h["neg_interro"]
                print(question_text,"tagtag",h["question_interro"],h["neg_interro"])
                if question_text[-1]=="?":
                    question_text=question_text[:-1].rstrip()
                question_text=" ".join([question_text,his_neg])
                paragraph["questions"][i]["input_text"]=question_text
                modify_count+=1
                print(question_text)
            count+=1

            """
                    if span_start!=-1:
                        sentence=answer_find(context_text,span_start,span_end,answer_replace)
                    else:
                        sentence=tf_idf(context_text," ".join(question_history[-2:]),num_canditate=1)
                        span_count+=1
                    sentence=modify(sentence,question_text)
                    sentence=" ".join(tokenize(sentence))
                    question_text=" ".join(tokenize(question_text))
                    sentences.append(sentence)
                    questions.append(question_text)
                else:
                    sentence=modify(context_text,question_text)
                    sentence=" ".join(tokenize(sentence))
                    question_text=" ".join(tokenize(question_text))
                    sentences.append(sentence)
                    questions.append(question_text)
            else:
                #break
                if len(question_history)>0:
                    q_his=question_history[-1]
                    sentence=modify_history(q_his,question_text)
                    sentence=" ".join(tokenize(sentence))
                    question_text=" ".join(tokenize(question_text))
                    sentences.append(sentence)
                    questions.append(question_text)
                else:
                    sentence=" ".join(tokenize(question_text))
                    question_text=" ".join(tokenize(question_text))
                    sentences.append(sentence)
                    questions.append(question_text)
                question_history.append(question_text)
            """

    print(count,modify_count)



    with open(output_path,"w")as f:
        json.dump(data,f)

#main
version="1.1"
type="interro"
question_modify=True
question_interro=True

data_process(input_path="data/coqa-dev-v1.0.json",
            output_path="data/coqa-dev-rule.json",
            dict_path="data/coqa-dev-corenlp.json",
            modify_path="data/pred_coqa-dev-{}.txt".format(type)
            )
"""
data_process(input_path="data/coqa-train-v1.0.json",
            output_path="data/coqa-train-modify.json",
            dict_path="data/coqa-train-corenlp.json",
            modify_path="data/pred_coqa-train-{}.txt".format(type)
            )
"""
