import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
from tqdm import tqdm
from collections import defaultdict
import collections
import math
from statistics import mean, median,variance,stdev


def compute_score(t,p):
    t_dict=collections.Counter(t)
    p_dict=collections.Counter(p)
    common=sum((t_dict & p_dict).values())
    return common

def compute_score_refs(ts,p):
    sum_dict=collections.Counter()
    p_dict=collections.Counter(p)
    for t in ts:
        t_dict=collections.Counter(t)
        sum_dict=(t_dict & p_dict) | sum_dict
    common=sum(sum_dict.values())
    return common

def n_gram(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def head_find(tokens):
    q_head=["what","how","who","when","which","where","why","whose","is","are","was","were","do","did","does"]
    for h in q_head:
        if h in tokens:
            return h
    return "<none>"

def head_compare(tgt,pred):
    t=head_find(tgt)
    p=head_find(pred)
    return t==p

modify=False

#tgt_path="tgt_test.txt"
#pred_path="pred_test.txt"
"""
if modify==False:
    src_path="../data/processed/src-dev.txt"
    tgt_path="../data/processed/tgt-dev.txt"
    pred_path="../data/pred.txt"
if modify==True:
    src_path="../data/processed/src-dev-modify.txt"
    tgt_path="../data/processed/tgt-dev.txt"
    pred_path="../data/pred_modify.txt"
"""

coqa=True

if coqa==False:
    type="interro_only"
    src_path="data/squad-src-dev-{}.txt".format(type)
    tgt_path="data/squad-tgt-dev-{}.txt".format(type)
    pred_path="data/pred_{}.txt".format(type)

if coqa==True:
    type="split2-interro"
    src_path="data/coqa-src-train-{}.txt".format(type)
    tgt_path="data/coqa-tgt-train-{}.txt".format(type)
    #pred_path="data/pred_coqa_{}.txt".format(type)
    pred_path="data/pred_coqa_split2_interro.txt"



src=[]
target=[]
predict=[]

"""
with open(src_path)as f:
    for line in f:
        src.append(line[:-1])

with open(tgt_path)as f:
    for line in f:
        target.append(line[:-1])

with open(pred_path)as f:
    for line in f:
        predict.append(line[:-1])
"""

with open(src_path)as f:
    for i,line in enumerate(f):
        src.append(line[:-1])

with open(tgt_path)as f:
    for i,line in enumerate(f):
        target.append(line[:-1])

with open(pred_path)as f:
    for i,line in enumerate(f):
        predict.append(line[:-1])



total=0
verb_count=0
if True:
    for i in tqdm(range(0,10000)):
        s=src[i]
        t=target[i]
        p=predict[i]
        if len(s)<100:
            print(s)
            print(t)
            print(p)
            print()

src2=[]

for s in src:
    #print(s)
    s=s.split()
    interro_pos=s.index("interro_tag")
    sent=s[interro_pos+1:]
    interro=s[:interro_pos]
    if interro[-1]=="?":
        interro=interro[:-1]
    sent=sent+interro
    src2.append(sent)
    #print(sent)

print(len(src),len(src2),len(target),len(predict))

target=[t.split() for t in target]
predict=[p.split() for p in predict]

"""
for i in tqdm(range(len(target))):
    s=src[i]
    t=target[i]
    p=predict[i]
    #"if len(t.split())<4:
    if corenlp.verb_check(t)==False:
        verb_count+=1
    total+=1
"""


#一文ずつ評価,corpusのサイズ考慮
if True:
    #nltk_src=src2
    nltk_target=[[t] for t in target]
    nltk_predict=predict
    print(nltk.translate.bleu_score.corpus_bleu(nltk_target,nltk_predict))
    print(nltk.translate.bleu_score.corpus_bleu(nltk_target,nltk_src))

    """
    count_target=sum(map(len,target))
    count_predict=sum(map(len,predict))
    penalty=math.exp(1-count_target/count_predict) if count_target>count_predict else 1
    print(penalty)
    bleu_score=[]

    for n in range(1,5):
        score_sum=0
        count_target=0
        count_predict=0
        for i in range(len(predict)):
            t=n_gram(target[i],n)
            p=n_gram(predict[i],n)
            score_sum+=compute_score(t,p)
            count_target+=len(t)
            count_predict+=len(p)
        bleu_score.append(score_sum/count_predict)
        score=penalty*math.exp(mean(map(math.log,bleu_score[0:n])))
        print("{}gram score is {}".format(n,score))
    """
"""
count=0
for t,p in zip(target,predict):
    count+=head_compare(t,p)

print(count/len(target))

target_dict=collections.Counter()
predict_dict=collections.Counter()

for s in target:
    target_dict[head_find(s)]+=1
print(target_dict.most_common())

for s in predict:
    predict_dict[head_find(s)]+=1
print(predict_dict.most_common())

count=0
for t,p in zip(target,predict):
    count+=head_compare(t,p)
print(count/len(target))
"""

"""
########################

#同じ文はまとめてtargetとして扱う。
#この手法は同じpredictについてもそれぞれ計算。元のはまとめて計算
#shortのoptionについてはほとんど一致
if False:
    print(len(src),len(target),len(predict))
    src=src[0:len(predict)]
    target=target[0:len(predict)]
    predict=predict[0:len(predict)]

    target_dict=defaultdict(lambda: [])
    predict_dict=defaultdict(str)
    src_set=set(src)

    for s,t,p in zip(src,target,predict):
        target_dict[s].append(t)
        predict_dict[s]=p

    print("size:{}\n".format(len(target_dict)))

    #calucurate penalty
    bleu_score=[]
    count_target=0
    count_predict=0
    for i,s in enumerate(src_set):
        t=target_dict[s]
        p=predict_dict[s]
        c_t=min(map(len,t))
        c_p=len(p)
        count_target+=c_t
        count_predict+=c_p

    penalty=math.exp(1-count_target/count_predict) if count_target>count_predict else 1
    print(penalty)
    print(count_target,count_predict)

    #n-gramごとにbleuを計算して平均を取る。
    #本家のbleuの計算はよくわからないので要検証
    for n in range(1,5):
        score_sum=0
        correct_count=0
        total_count=0
        for i,s in enumerate(src_set):
            t=[n_gram(sent,n) for sent in target_dict[s]]
            p=n_gram(predict_dict[s],n)
            correct_num=compute_score_refs(t,p)
            correct_count+=correct_num
            total_count+=len(p)
        print(correct_count,total_count)
        score=correct_count/total_count
        bleu_score.append(score)
        score=penalty*math.exp(mean(map(math.log,bleu_score[0:n])))
        print("{}gram score is {}".format(n,score))
        print()
"""
######################################
"""
#一文ずつ評価(sentence_bleu使用)
if False:
    score_sum_bleu1=0
    score_sum_bleu2=0
    for t,p in tqdm(zip(target,predict)):
        t=word_tokenize(t)
        p=word_tokenize(p)
        score = sentence_bleu([t],p,weights=(1,0,0,0))
        score_sum_bleu1+=score
        score = sentence_bleu([t],p,weights=(0,1,0,0))
        score_sum_bleu2+=score

    print(score_sum_bleu1/len(target),len(target))
    print(score_sum_bleu2/len(target),len(target))

"""
