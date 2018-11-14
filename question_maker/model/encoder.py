import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.vocab_size = args.vocab_size
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.batch_size=0
        self.hidden=0

        self.word_embed=nn.Embedding(self.vocab_size, self.embed_size,padding_idx=0,
                                    _weight=torch.from_numpy(args.pretrained_weight).float())
        self.gru=nn.GRU(self.embed_size,self.hidden_size,bidirectional=True,dropout=args.dropout,batch_first=True)

    def forward(self,input):#input:(batch,seq_len)
        embed = self.word_embed(input)#(batch,seq_len,embed_size)
        output, hidden = self.gru(embed)#(batch,seq_len,hidden_size*2),(1,batch,hidden_size*2)
        print(output.size(),hidden.size())
        return output, hidden[0]
