#SQuADのデータ処理

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


def data_process(input_path,output_path,word_count,lower=True):
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
            if lower:
                context_text=context_text.lower()
            context=tokenize(context_text)
            context.append("<eos>")
            #メモリサイズの確保のため、サイズが大きいcontextはスキップ
            #結果的に、87599個のcontextのうち、500をカット
            if len(context)>=350:
                continue
            if word_count:
                for word in context:
                    word2count[word]+=1
                    for char in word:
                        char2count[char]+=1
            for qas in paragraph["qas"]:
                question_text=qas["question"]
                if lower:
                    question_text=question_text.lower()
                if len(qas["answers"])==0:
                    continue
                question=tokenize(question_text)
                question.append("<eos>")
                if word_count:
                    for word in question:
                        word2count[word]+=1
                        for char in word:
                            char2count[char]+=1

                contexts.append(context)
                questions.append(question)
                id_num=qas["id"]
                ids.append(id_num)
                a=qas["answers"][0]
                answer_start=a["answer_start"]
                answer_end=a["answer_start"]+len(a["text"])
                answer=tokenize(a["text"])
                answer.append("<eos>")
                answer_sent=answer_find(context_text,answer_start,answer_end)#contextの中からanswerが含まれる文を見つけ出す
                answer_sent.append("<eos>")
                answer_start,answer_end=c2wpointer(context_text,context,answer_start,answer_end)
                answer_starts.append(answer_start)
                answer_ends.append(answer_end)
                answers.append(answer)
                sentences.append(answer_sent)

    with open(output_path,"w")as f:
        t={"contexts":contexts,
            "questions":questions,
            "answer_starts":answer_starts,
            "answer_ends":answer_ends,
            "answers":answers,
            "sentences":sentences,
            "ids":ids}
        json.dump(t,f)

    if word_count:
        #<pad>:0,<unk>:1,<eos>:2
        #pad:padding用,unk:unknownトークン,eos:End of Sentence

        word2id={w:i for i,(w,count) in enumerate(word2count.items(),3) if count>=0}
        word2id["<pad>"]=0
        word2id["<unk>"]=1
        word2id["<eos>"]=2
        char2id={c:i for i,(c,count) in enumerate(char2count.items(),2) if count>=0}
        char2id["<eos>"]=0
        char2id["<unk>"]=1
        vec_process(contexts,word2id,char2id)


def vec_process(contexts,word2id,char2id):
    vec_size=100

    #vec==300は単語が空白で区切られているものがあり、要対処
    if vec_size==300:
            path="data/glove.840B.300d.txt"
    else:
        path="data/glove.6B.{}d.txt".format(vec_size)
    w2vec={}
    id2vec=np.zeros((len(list(word2id.items())),vec_size))

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
data_process(input_path="data/squad_train-v{}.json".format(version),output_path="data/squad_train_data.json",word_count=True,lower=True)
data_process(input_path="data/squad_dev-v{}.json".format(version),output_path="data/squad_test_data.json",word_count=False,lower=True)
