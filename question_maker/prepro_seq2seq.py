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
import random

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

def history_maker(question_interro,neg_interro):
    if question_interro !="no_interro":
        question=question_interro+" "+neg_interro
        return question
    else:
        interro_list=["what","where","who","why","which","whom","how"]
        index=random.randrange(len(interro_list))
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



def data_process(input_path,src_path,tgt_path,question_modify,train=True,complete=True,paragraph=True,history=True):
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
    corenlp=CoreNLP()
    count=0
    pos_interro="what"#擬似的な一つ前の疑問詞
    for topic in tqdm(data["data"]):
        topic=topic["paragraphs"]
        for paragraph in topic:
            context_text=paragraph["context"].lower()
            question_history=[]
            for qas in paragraph["qas"]:
                question_text=qas["question"].lower()
                if len(qas["answers"])==0:
                    continue
                a=qas["answers"][0]
                answer_text=a["text"].lower()
                answer_start=a["answer_start"]
                answer_end=a["answer_start"]+len(a["text"])
                sentence=answer_find(context_text,answer_start,answer_end,answer_replace)#contextの中からanswerが含まれる文を見つけ出す
                if len(question_text)<=5:#ゴミデータ(10個程度)は削除
                    continue
                if question_modify==True:
                    if history==False:
                        question_interro,neg_interro=corenlp.forward(question_text)#疑問詞を探してくる
                        if question_interro=="none_tag":
                            continue
                        sentence_interro=modify(context_text,question_interro)
                        sentence_interro=" ".join(tokenize(sentence_interro))
                        answer_text=" ".join(tokenize(answer_text))
                        question_text=" ".join(tokenize(question_text))
                        questions.append(question_text)
                        sentences.append(sentence_interro)
                        if complete==True:#省略していないものを含める
                            sentence=modify(context_text,question_text)
                            sentence=" ".join(tokenize(sentence))
                            answer_text=" ".join(tokenize(answer_text))
                            question_text=" ".join(tokenize(question_text))
                            questions.append(question_text)
                            sentences.append(sentence)
                    else:
                        question_interro,neg_interro=corenlp.forward(question_text)#疑問詞を探してくる
                        if question_interro=="none_tag":
                            continue
                        if len(question_history)>0:#histroyがある場合
                            q_his,q_his_interro=question_history[-1]
                            if question_interro.rstrip()[-1]!="?":
                                question_interro_x=question_interro+" ?"
                            #疑問詞からhistoryを制作
                            history=history_maker(q_his_interro,neg_interro)
                            sentence=modify_history(history,question_interro_x)
                            sentence=" ".join(tokenize(sentence))
                            question_text=" ".join(tokenize(question_text))
                            sentences.append(sentence)
                            questions.append(question_text)
                            #前の文をそのまま利用
                            sentence2=modify_history(q_his,question_text)
                            sentence2=" ".join(tokenize(sentence2))
                            sentences.append(sentence2)
                            questions.append(question_text)
                        else:#historyがない、最初の位置の場合
                            #historyなしの場合
                            sentence=" ".join(tokenize(question_text))
                            question_text=" ".join(tokenize(question_text))
                            sentences.append(sentence)
                            questions.append(question_text)
                            #疑問詞だけでhistroyを作成
                            history=history_maker(pos_interro,"?")
                            sentence2=modify_history(history,question_text)
                            sentence2=" ".join(tokenize(sentence2))
                            sentences.append(sentence2)
                            questions.append(question_text)

                        question_history.append((question_text,question_interro))
                        pos_interro=question_interro


    print(len(questions),len(sentences))

    with open(src_path,"w")as f:
        for s in sentences:
            f.write(s+"\n")

    with open(tgt_path,"w")as f:
        for s in questions:
            f.write(s+"\n")

#main
version="1.1"
type="interro_history"
question_modify=True
question_interro=True

random.seed(0)


data_process(input_path="data/squad-train-v{}.json".format(version),
            src_path="data/squad-src-train-{}.txt".format(type),
            tgt_path="data/squad-tgt-train-{}.txt".format(type),
            question_modify=True,
            train=True,
            complete=True,
            paragraph=False,
            history=True
            )
"""
data_process(input_path="data/squad-dev-v{}.json".format(version),
            src_path="data/squad-src-dev-{}.txt".format(type),
            tgt_path="data/squad-tgt-dev-{}.txt".format(type),
            question_modify=True,
            train=False,
            complete=True,
            paragraph=False,
            history=True
            )
"""
