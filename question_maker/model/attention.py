import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, args):
        super(Attention, self).__init__()

        self.hidden_size = args.hidden_size
        self.attention_wight=nn.Linear(self.hidden_size,self.hidden_size*2)

    def forward(self,input,encoder_output):#input:(batch,hidden_size),encoder_input:(batch,seq_len,hidden_size*2)
        #input*W*encoder_input:(batch,seq_len)(batch,1,hidden_size)*(hidden_size,hidden_size*2)*(batch,seq_len,hidden_size*2)
        attention_input=torch.unsqueeze(input,1)#(batch,1,hidden_size)
        attention_encoder_input=torch.transpose(encoder_output,1,2)#(batch,hidden_size*2,seq_len)
        attention_output=F.softmax(torch.bmm(self.attention_wight(attention_input),attention_encoder_input),dim=-1)#(batch,1,seq_len)
        attention_output=torch.squeeze(torch.bmm(attention_output,encoder_output),1)#(batch,hidden_size*2)
        return attention_output
