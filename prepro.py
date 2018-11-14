#coqaのデータ構築

import json
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
import pickle
import collections
import string
import re

#sをnormalizeして返す、Squadのevalのもの
def normalize_answer(s):
    """Lower text and remove punctuation, storys and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

#offsetsからstart-endの範囲を探して単語単位のidで返す
def find_span(offsets, start, end):
    start_index = end_index = -1
    for i, offset in enumerate(offsets):
        if (start_index < 0) or (start >= offset[0]):
            start_index = i
        if (end_index < 0) and (end <= offset[1]):
            end_index = i
    return (start_index, end_index)

#f1スコアが一番高いものをcontextから探して返す
def find_span_with_gt(context, offsets, ground_truth):
    best_f1 = 0.0
    best_span = (len(offsets) - 1, len(offsets) - 1)
    gt = normalize_answer(ground_truth).split()

    ls = [i for i in range(len(offsets))
          if context[offsets[i][0]:offsets[i][1]].lower() in gt]

    for i in range(len(ls)):
        for j in range(i, len(ls)):
            pred = normalize_answer(context[offsets[ls[i]][0]: offsets[ls[j]][1]]).split()
            common = collections.Counter(pred) & collections.Counter(gt)
            num_same = sum(common.values())
            if num_same > 0:
                precision = 1.0 * num_same / len(pred)
                recall = 1.0 * num_same / len(gt)
                f1 = (2 * precision * recall) / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_span = (ls[i], ls[j])
    return best_span

#tokensを取って、それぞれのtokenのoffsetsを返す
#ex. Hey me->{(1,4),(5,6)}
def offsets_process(text,tokens):
    token_ids=[]
    cur_id=0
    for i,token in enumerate(tokens):
        start=text.find(token,cur_id)
        token_ids.append((start,start+len(token)))
        cur_id=start+len(token)
    return token_ids

#sentenceを受け取り、tokenizeして返す
def tokenize(sent):
    return [token.replace('``','"').replace("''",'"') for token in nltk.word_tokenize(sent)]

#trainデータの処理とtrainデータを使ってw2vecも作る
def data_process(input_path,output_path,word_count):
    with open(input_path,"r") as f:
        data=json.load(f)
    contexts=[]
    questions=[]
    answer_starts=[]
    answer_ends=[]
    answer_texts=[]
    ids=[]
    turn_ids=[]
    word2count=collections.Counter()
    char2count=collections.Counter()

    for paragraph in tqdm(data["data"]):
        context_text=paragraph["story"]
        context=tokenize(context_text)
        id_num=paragraph["id"]
        offsets=offsets_process(context_text,context)
        if word_count:
            for word in context:
                word2count[word]+=1
                for char in word:
                    char2count[char]+=1
        for i in range(len(paragraph["questions"])):
            question_dict=paragraph["questions"][i]
            answer_dict=paragraph["answers"][i]
            question_text=question_dict["input_text"]
            question=tokenize(question_text)
            answer_text=answer_dict["input_text"]
            span_start=answer_dict["span_start"]
            span_end=answer_dict["span_end"]
            span_text=answer_dict["span_text"]
            turn_id=paragraph["questions"][i]["turn_id"]

            #answerがspanの中にある場合はそのindexを調べる
            if answer_text in span_text:
                i=span_text.find(answer_text)
                answer_start,answer_end=find_span(offsets,span_start+i,span_start+i+len(answer_text))
            #ない場合はcontextの全てのspanとgtを比較して一番f1が高いspanをgtとして、そのspanのidを返す
            else:
                answer_start,answer_end=find_span_with_gt(context_text,offsets,answer_text)

            contexts.append(context)
            questions.append(question)
            answer_starts.append(answer_start)
            answer_ends.append(answer_end)
            answer_texts.append(answer_text)
            ids.append(id_num)
            turn_ids.append(turn_id)

            if word_count:
                for word in question:
                    word2count[word]+=1
                    for char in word:
                        char2count[char]+=1

    with open(output_path,"wb")as f:
        t={"contexts":contexts,
            "questions":questions,
            "answer_starts":answer_starts,
            "answer_ends":answer_ends,
            "answer_texts":answer_texts,
            "id":id,
            "turn_id":turn_ids}
        json.dump(t,f)

    if word_count:
        #<eos>:0,<unk>:1
        word2id={w:i for i,(w,count) in enumerate(word2count.items(),2) if count>=0}
        word2id["<eos>"]=0
        word2id["<unk>"]=1
        char2id={c:i for i,(c,count) in enumerate(char2count.items(),2) if count>=0}
        char2id["<eos>"]=0
        char2id["<unk>"]=1
        #vec_process(contexts,word2id,char2id)

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

    with open("data/word2id2vec.pickle","wb")as f:
        t={"word2id":word2id,
            "id2vec":id2vec,
            "char2id":char2id}
        json.dump(t,f)

#main
#data_process(input_path="data/coqa-train.json",output_path="data/train_data.json",word_count=True)
data_process(input_path="data/coqa-dev.json",output_path="data/test_data.json",word_count=False)
