"""Official evaluation script for CoQA.

The code is based partially on SQuAD 2.0 evaluation script.
"""
import argparse
import json
import re
import string
import sys

from collections import Counter, OrderedDict
from nltk.tokenize import word_tokenize,sent_tokenize

def head_find(tgt):
    q_head=["what","how","who","when","which","where","why","whose","whom","is","are","was","were","do","did","does"]
    tgt_tokens=word_tokenize(tgt)
    true_head="<none>"
    for h in q_head:
        if h in tgt_tokens:
            true_head=h
            break
    return true_head

with open("data/pipeline.prediction_his_1.json")as f:
    pred1=json.load(f)

with open("data/coqa-dev-v1.0.json")as f:
    data=json.load(f)

with open("data/coqa-dev-corenlp.json")as f:
    corenlp_data=json.load(f)

with open("data/pipeline.prediction_interro_his_0.json")as f:
    pred2=json.load(f)

modify=[]
with open("data/pred_coqa-dev-interro.txt")as f:
    for line in f:
        modify.append(line.rstrip())

corenlp_count=0
ans_count=0
mod_count=0

text=json.dumps(corenlp_data[0:100],indent=4)
mydict=Counter()

gold_dict = {}
id_to_source = {}
for story in data['data']:
    source = story['source']
    story_id = story['id']
    questions = story['questions']
    multiple_answers = [story['answers']]
    multiple_answers += story['additional_answers'].values()
    for i, qa in enumerate(questions):
        c_data=corenlp_data[corenlp_count]
        if c_data["vb_check"]==False and c_data["question_interro"]!="none_tag":
            qid = qa['turn_id']
            if i + 1 != qid:
                sys.stderr.write("Turn id should match index {}: {}\n".format(i + 1, qa))
            gold_answers = []
            for answers in multiple_answers:
                answer = answers[i]
                if qid != answer['turn_id']:
                    sys.stderr.write("Question turn id does match answer: {} {}\n".format(qa, answer))
                gold_answers.append(answer['input_text'])
            key = (story_id, qid)
            if key in gold_dict:
                sys.stderr.write("Gold file has duplicate stories: {}".format(source))
            gold_dict[key] = gold_answers
            q_text=qa["input_text"].lower()
            interro=head_find(q_text)

            mydict[interro]+=1


            mod_count+=1
        corenlp_count+=1
for k,v in mydict.items():
    print(k,v/sum(mydict.values()))
print(sum(mydict.values()))
