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

from func.corenlp import CoreNLP



def data_process(input_path,pred_path,modify_path):
    with open(input_path,"r") as f:
        data=json.load(f)

    pred=[]
    with open(pred_path,"r") as f:
        for line in f:
            pred.append(line.rstrip())

    count=0
    dif_count=0
    verb_count=0

    corenlp=CoreNLP()
    for paragraph in tqdm(data["data"]):
        for i in range(len(paragraph["questions"])):
            question_text=paragraph["questions"][i]["input_text"].lower()
            print(question_text)
            print(pred[count])
            print()
            if question_text!=pred[count]:
                #print(question_text)
                if corenlp.verb_check(question_text)==False:
                    paragraph["questions"][i]["input_text"]=pred[count]
                    verb_count+=1
                dif_count+=1
            count+=1

    print(count,dif_count,verb_count)
"""
    with open(modify_path,"w")as f:
        json.dump(data,f)
"""


data_process(input_path="data/coqa-train-v1.0-split2.json",
            pred_path="data/pred_coqa_split2_interro.txt",
            modify_path="data/coqa-train-v1.0-split2-modify-verb.json",
            )
"""
data_process(input_path="data/coqa-dev.json",
            src_path="data/coqa-src-dev-{}.txt".format(type),
            tgt_path="data/coqa-tgt-dev-{}.txt".format(type),
            question_modify=True,
            train=False,
            sub=True,
            paragraph=False,
            history=True
            )
"""
