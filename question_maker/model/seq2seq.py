import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from question_maker.model.encoder import Encoder
from question_maker.model.decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self,args):
        super(Seq2Seq, self).__init__()
        self.encoder=Encoder(args)
        self.decoder=Decoder(args)

    def forward(self, context_words,train=True):
        encoder_outputs, encoder_hidden = self.encoder(context_words)
        output = self.decoder(encoder_outputs,encoder_hidden)
        return output
