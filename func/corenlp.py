from pycorenlp import StanfordCoreNLP
from tqdm import tqdm

import matplotlib.pyplot as plt
import networkx as nx


#corenlpを用いて疑問詞とその周辺を取り出す
#ex. How many icecream did you eat? -? How many icecream
#疑問詞を特定できないもの(interro_listの中のタグがあわられないもの)は除去


class CoreNLP():
    def __init__(self):
        self.nlp=StanfordCoreNLP('http://localhost:9000')
        self.interro_list=["WDT","WP"," WP$","WRB","VB","VBD","VBG","VBN","VBP","VBZ"]
        self.count=-1

    def forward(self,text):#input:(batch,seq_len)
        self.count+=1
        q=self.nlp.annotate(text, properties={'annotators': 'tokenize,ssplit,parse','outputFormat': 'json'})

        tokens=q["sentences"][0]["tokens"]
        deps=q["sentences"][0]["basicDependencies"]

        token_list=[]
        token_list.append({"index":0,"text":"ROOT"})
        interro_id=-1
        for token in tokens:
            token_list.append({"index":token["index"],"text":token["originalText"],"pos":token["pos"]})
        for token in tokens:
            if interro_id==-1 and token["pos"] in self.interro_list[0:4]:#疑問詞のチェック
                interro_id=token["index"]
        for token in tokens:
            if interro_id==-1 and token["pos"] in self.interro_list[4:]:#疑問詞のチェック
                interro_id=token["index"]

        #疑問詞がなかった時のエラー処理
        if interro_id==-1:
            print(self.count)
            return "none_tag"


        G = nx.DiGraph()
        G.add_nodes_from(range(len(token_list)))
        for dep in deps:
            G.add_path([dep["dependent"],dep["governor"]])
        if nx.has_path(G,interro_id,0)==False:
            print("error")
        s_path=nx.shortest_path(G,interro_id,0)


        if len(s_path)==2:#疑問詞だけ
            node_list=[s_path[0]]
        else:#疑問詞周り
            node_list=[node for node in G.nodes() if nx.has_path(G,node,s_path[-3])]
        question=" ".join([token_list[node]["text"] for node in node_list])

        return question
