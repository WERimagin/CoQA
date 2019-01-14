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

from func.tf_idf import tf_idf,cos_sim


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

    return answer_sent,sent_start_id


def data_process(input_path,src_path,tgt_path,dict_path,test=True):
    with open(input_path,"r") as f:
        data=json.load(f)
    with open(dict_path,"r") as f:
        corenlp_data=json.load(f)
    contexts=[]
    questions=[]
    answer_starts=[]
    answer_ends=[]
    answer_texts=[]
    answers=[]
    sentences=[]
    ids=[]
    answer_replace=False
    count=-1
    answer_count=[]


    path="/home/6/15B06641/data/glove.840B.300d.txt"

    w2vec={}

    with open(path,"r")as f:
        print(path)
        for i,line in tqdm(enumerate(f)):
            line_split=line.split()
            w2vec[" ".join(line_split[0:-300])]=[float(i) for i in line_split[-300:]]
            if i==50000:
                break

    print("end")



    for paragraph in tqdm(data["data"]):
        context_text=paragraph["story"].lower()
        question_history=[]
        for i in range(len(paragraph["questions"])):
            count+=1

            question_dict=paragraph["questions"][i]
            answer_dict=paragraph["answers"][i]
            question_text=question_dict["input_text"].lower()
            answer_text=answer_dict["input_text"].lower()
            question_history.append((question_text,answer_text))

            span_start=answer_dict["span_start"]
            span_end=answer_dict["span_end"]
            span_text=answer_dict["span_text"]
            turn_id=paragraph["questions"][i]["turn_id"]

            d=corenlp_data[count]
            if d["vb_check"]==False and d["question_interro"]!="none_tag":
                if test==False:
                    start=0
                    sentence=tf_idf(context_text," ".join(question_history[-2:]),num_canditate=1)
                    if span_start!=-1:
                        sentence=answer_find(context_text,span_start,span_end,answer_replace)
                else:
                    if span_start==-1:
                        continue
                    start=0
                    if len(question_history)>=2:
                        join_text=" ".join([question_history[-2][0],question_history[-2][1],question_history[-1][0]])
                    else:
                        join_text=question_history[-1][0]
                    sentence,answer_id=answer_find(context_text,span_start,span_end,answer_replace)


                    para_vec=[np.sum([w2vec[word] for word in word_tokenize(s) if word in w2vec])
                                for s in sent_tokenize(context_text)]

                    sent_vec=np.sum([w2vec[word] for word in word_tokenize(join_text) if word in w2vec])

                    cos={i:cos_sim(v,sent_vec) for i,v in enumerate(para_vec)}
                    cos=sorted(cos.items(),key=lambda x:-x[1])
                    tf_id=[c[0] for c in cos[0:1]]

                    answer_count.append(answer_id in tf_id)


                sentence=modify(sentence,question_text)
                sentence=" ".join(tokenize(sentence))
                question_text=" ".join(tokenize(question_text))
                sentences.append(sentence)
                questions.append(question_text)

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

    print(np.average(answer_count),len(answer_count))
    print(len(questions),len(sentences))




#main
version="1.1"
type="interro_cand1"
question_modify=True
question_interro=True

"""
data_process(input_path="data/coqa-train-v1.0.json",
            src_path="data/coqa-src-train-{}.txt".format(type),
            tgt_path="data/coqa-tgt-train-{}.txt".format(type),
            dict_path="data/coqa-train-corenlp.json"
            )
"""
data_process(input_path="data/coqa-dev-v1.0.json",
            src_path="data/coqa-src-dev-{}.txt".format(type),
            tgt_path="data/coqa-tgt-dev-{}.txt".format(type),
            dict_path="data/coqa-dev-corenlp.json",
            test=True
            )
