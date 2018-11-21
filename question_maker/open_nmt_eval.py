import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

#tgt_path="tgt_test.txt"
#pred_path="pred_test.txt"
tgt_path="data/pred.txt"
pred_path="data/squad-tgt-val.txt"

target=[]
predict=[]

with open(tgt_path)as f:
    for line in f:
        target.append(line)

with open(pred_path)as f:
    for line in f:
        predict.append(line)

target=[word_tokenize(sent) for sent in target]
predict=[word_tokenize(sent) for sent in predict]

for i in range(10):
    t=target[i]
    p=predict[i]
    print(t,p)
    print(sentence_bleu([t],p))


score_sum=0
for t,p in tqdm(zip(target,predict)):
    score = sentence_bleu([t],p)
    score_sum+=score

print(score_sum/len(target),len(target))
