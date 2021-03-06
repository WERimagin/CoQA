import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from func.char_embedding import Char_Embedding
from func.highway_net import Highway_Net

class Bidaf(nn.Module):
    def __init__(self,args):
        super(Bidaf, self).__init__()
        self.vocab_size = args.vocab_size
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.batch_size=0
        self.hidden=0

        self.word_embed=nn.Embedding(self.vocab_size, self.embed_size,padding_idx=0,_weight=torch.from_numpy(args.pretrained_weight).float())
        self.char_embed=Char_Embedding(args)
        self.highway_net=Highway_Net(args)
        self.lstm_layer=nn.GRU(self.embed_size*2,self.hidden_size,bidirectional=True,dropout=args.dropout,batch_first=True)

        self.embed_W=nn.Linear(self.hidden_size*6,1)
        self.modeling_layer=nn.GRU(self.hidden_size*8,self.hidden_size,bidirectional=True,dropout=args.dropout,batch_first=True)
        self.p1_W=nn.Linear(self.hidden_size*10,1)
        self.p2_W=nn.Linear(self.hidden_size*10,1)
        self.p2_lstm=nn.GRU(self.hidden_size*2,self.hidden_size,bidirectional=True,dropout=args.dropout,batch_first=True)

        self.dropout=nn.Dropout(args.dropout)

        #self.p3_b=torch.from_numpy()


    def embedding(self,words,chars):
        embed_w=self.word_embed(words)
        embed_c=self.char_embed(chars)
        x=torch.cat((embed_w,embed_c),2)#(N,T,embed_size*2)
        x=self.highway_net(x)#(N,T,embed_size*2)
        x=self.lstm_layer(x)[0]#(N,T,2d)
        return x


    #メイン処理層
    def __call__(self, context_words,query_words,context_chars,query_chars,train=True):#batch*len(sentense)のデータをbatch*1として切り出してlesmにforループで渡す
        self.batch_size=context_words.size(0)
        N=self.batch_size
        T=context_words.size(1)
        J=query_words.size(1)

        #1~3
        embed_context=self.embedding(context_words,context_chars) #(N,T,2d)
        embed_query=self.embedding(query_words,query_chars)       #(N,J,2d)

        #4
        embed_context_ex=embed_context.view(N,T,1,self.hidden_size*2).expand(N,T,J,self.hidden_size*2)#(N,T,J,2d)
        embed_query_ex=embed_query.view(N,1,J,self.hidden_size*2).expand(N,T,J,self.hidden_size*2)#(N,T,J,2d)
        embed_mul=torch.mul(embed_context_ex,embed_query_ex)#(N,T,J,2d)
        embed_cat=torch.cat((embed_context_ex,embed_query_ex,embed_mul),-1)#(N,T,J,6d)
        S=self.embed_W(embed_cat).view(N,T,J)#(N,T,J)


        c2q=torch.bmm(F.softmax(S,dim=-1),embed_query)#(N,T,2d)#bmm:batchを考慮した行列の積?
        q2c=torch.bmm(F.softmax(torch.max(S,dim=-1)[0],-1).view(N,1,T),embed_context)#(N,1,2d)=(N,1,T)*(N,T,2d)
        q2c=q2c.repeat(1,T,1)#(N,T,2d)

        embed_mul1=torch.mul(embed_context,c2q)
        embed_mul2=torch.mul(embed_context,q2c)
        G=F.relu(torch.cat((embed_context,c2q,embed_mul1,embed_mul2),-1))#(N,T,8d)#self-attentionのを参考にreluを追加

        #5
        M=self.modeling_layer(G)[0]#(N,T,2d)

        #6
        GM=torch.cat((G,M),-1)#(N,T,10d)
        p1=self.dropout(self.p1_W(GM)).view(N,T)#(N,T)
        M2=self.p2_lstm(M)[0]#(N,T,2d)
        GM=torch.cat((G,M2),-1)#(N,T,10d)
        p2=self.dropout(self.p2_W(GM)).view(N,T)#(N,T)
        return p1,p2
