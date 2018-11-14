import torch
import torch.nn as nn
import torch.nn.functional as F


class Char_Embedding(nn.Module):


    def __init__(self, args):
        super(Char_Embedding, self).__init__()
        self.c_embed_size=args.c_embed_size
        self.embedding=nn.Embedding(args.c_vocab_size, args.c_embed_size)
        self.conv=nn.Conv1d(args.c_embed_size,args.embed_size,5)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self,x):
        #x:(N,sentence_size,word_size)
        #まずembeddingでcharをベクトルへ
        N=x.size(0)
        sentence_size=x.size(1)
        word_size=x.size(2)

        #embedding
        x=x.view(N*sentence_size,word_size)#(N*sentence_size,word_size,1)
        x=self.embedding(x)#(N*sentence_size,word_size,c_embed_size)
        x=torch.transpose(x,1,2)#(N*sentence_size,c_embed_size,word_size)channnelをc_embed_sizeにするため

        #convとmaxpooling
        x=F.relu(self.conv(x))#(N*sentence_size,w_embed_size,word_size-4)
        x=torch.max(x,dim=-1)[0]#(N*sentence_size,w_embed_size)
        x=x.view(N,sentence_size,-1)
        x=self.dropout(x)

        return x
