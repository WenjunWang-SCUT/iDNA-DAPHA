import torch
import torch.nn as nn
# import json
# import torch.nn.functional as F
# from pykalman import KalmanFilter

from model import DNAbert
# from model.pytorch_revgrad.module import RevGrad

'''bert Fusion 模型'''
class FusionBERT(nn.Module):
    def __init__(self, config):
        super(FusionBERT, self).__init__()
        self.config = config

        self.config.kmer = self.config.kmers[0]
        self.bertone = DNAbert.BERT(self.config)

        self.config.kmer = self.config.kmers[1]
        self.berttwo = DNAbert.BERT(self.config)
        
        self.classification = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, seqs):
        # print(seqs)
        representationX = self.bertone(seqs)
        representationY = self.berttwo(seqs)

        representation = torch.cat((representationX, representationY), dim=1)
        # representation = representationX + representationY

        output = self.classification(representation)
        return output, representation
