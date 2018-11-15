import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, args):
        super(Attention, self).__init__()

        self.hidden_size = args.hidden_size
        self.attention_wight=nn.Linear(self.hidden_size,self.hidden_size*2)

    def forward(self,input,encoder_output):#input:(batch,hidden_size),encoder_input:(batch,seq_len,hidden_size*2)
        #input*W*encoder_input:(batch,seq_len)(batch,hidden_size)*(W)*(batch,seq_len,hidden_size*2)
        attention_output=F.softmax(torch.bmm(self.attention_wight(input),encoder_output),dim=-1)#(batch,seq_len)
        attention_output=torch.bmm(attention_output,encoder_output)#(batch,hidden_size*2)
        return attention_output
