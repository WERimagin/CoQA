import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from func.utils import Word2Id,DataLoader,make_vec,make_vec_c,to_var
from question_maker.model.attention import Attention

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
        self.gru=nn.GRU(self.embed_size,self.hidden_size,bidirectional=False,dropout=args.dropout,batch_first=True)#decoderは双方向にできない


        self.attention=Attention(args)
        self.attention_wight=nn.Linear(self.hidden_size*3,self.hidden_size*3)
        self.out=nn.Linear(self.hidden_size*3,self.vocab_size)

    def decode_step(self,input,encoder_output):#(batch,1)
        embed=self.word_embed(input)#(batch,1,embed_size)
        output,hidden=self.gru(embed,self.hidden)#(batch,1,hidden_size)
        self.hidden=hidden
        output=torch.squeeze(1)#(batch,hidden_size)
        attention_output=self.attention(output,encoder_output)#(batch,hidden_size*2)

        output=F.relu(self.attention(torch.cat((output,attention_output),-1))))#(batch,hidden_size*3)
        output=self.out(output)#(batch,vocab_size)
        predict=torch.argmax(output,dim=-1) #(batch)
        return output,predict

    def forward(self,encoder_output,encoder_hidden,q_words):#input:(batch,seq_len),encoder_hidden:(seq_len,hidden_size*2),q_words:(batch,q_seq_len)
        batch_size=q_words.size(0)
        q_seq_len=q_words.size(1)
        self.hidden=encoder_hidden
        outputs=to_var(torch.from_numpy(np.zeros((q_seq_len,batch_size,self.vocab_size))))
        current_input=to_var(torch.from_numpy(np.zeros((batch_size,1),dtype="long")))#最初の隠れベクトル,<SOS>

        for i in range(q_seq_len):#(batch,1)
            output,predict=self.decode_step(current_input,encoder_output)#(batch,vocab_size)(batch)
            current_input=predict.view(-1,1)
            outputs[i]=output

        outputs=torch.transpose(outputs,0,1)#(batch,seq_len,vocab_size)
        return outputs
