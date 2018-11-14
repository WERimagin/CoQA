import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from func.utils import Word2Id,DataLoader,make_vec,make_vec_c,to_var

class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.vocab_size = args.vocab_size
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.batch_size=0
        self.hidden=0

        self.word_embed=nn.Embedding(self.vocab_size, self.embed_size,padding_idx=0,
                                    _weight=torch.from_numpy(args.pretrained_weight).float())
        self.gru=nn.GRU(self.embed_size,self.hidden_size,bidirectional=True,dropout=args.dropout,batch_first=True)
        self.out=nn.Linear(self.hidden_size*2,self.vocab_size)


    def decode_step(self,input):#(batch,1)
        embed=self.word_embed(input)#(batch,1,embed_size)
        print(input.size(),embed.size())
        output,hidden=self.gru(embed,self.hidden)#(batch,1,hidden_size*2)
        self.hidden=hidden
        output=torch.squeeze(self.out(output),1)#(batch,vocab_size)
        predict=torch.argmax(output,dim=-1)[0]#(batch)
        return output,predict

    def forward(self,input,encoder_hidden):#input:(batch,seq_len),encoder_hidden:(seq_len,hidden_size*2)
        batch_size=input.size(0)
        seq_len=input.size(1)
        self.hidden=encoder_hidden
        outputs=to_var(torch.from_numpy(np.zeros((seq_len,batch_size,self.vocab_size))))

        current_input=to_var(torch.from_numpy(np.zeros((batch_size,1),dtype="long")))
        for i in range(seq_len):#(batch,1)
            output,predict=self.decode_step(current_input)#(batch,vocab_size)(batch)
            current_input=predict
            outputs[i]=output

        outputs=torch.transpose(outputs,(0,1))#(batch,seq_len,vocab_size)
        return outputs
