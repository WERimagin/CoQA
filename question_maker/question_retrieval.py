from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

question_list=[]
input_path="data/squad-src-train-normal.txt"


with open(input_path) as f:
    for line in f:
        question_list=line[:-1]

for i,q in enumerate(question_list):
    if i>=100:
        break
    text=q
    output=nlp.annotate(text, properties={'annotators': 'tokenize,ssplit,parse','outputFormat': 'json'})
    print(output)
