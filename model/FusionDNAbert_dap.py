import torch
import torch.nn as nn

from model import DNAbert
from model.pytorch_revgrad.module import RevGrad

'''bert Fusion 模型'''
class FusionBERT(nn.Module):
    def __init__(self, config):
        super(FusionBERT, self).__init__()
        self.config = config

        self.config.kmer = self.config.kmers[0]
        self.bertone = DNAbert.BERT(self.config)

        self.config.kmer = self.config.kmers[1]
        self.berttwo = DNAbert.BERT(self.config)

        self.classification_4mCF = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_4mCS = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_4mCC = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_4mCT = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_5hmCH = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_5hmCM = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_6mAA = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_6mACEL = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_6mACEQ = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_6mAD = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_6mAF = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_6mAH = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_6mAR = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_6mAS = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_6mATT = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_6mATO = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.classification_6mAX = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

        self.fc_task = nn.Sequential(
            RevGrad(alpha=0.3),
            nn.Linear(768 * 2, 128),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(128, 17)
        )

    def forward(self, seqs, type):
        # print(seqs)
        representationX = self.bertone(seqs)
        representationY = self.berttwo(seqs)

        # print(representationX)
        # print(representationY)

        # F = torch.sigmoid(self.Ws * representationX + self.Wh * representationY)
        # representation = F * representationX + (1 - F) * representationY

        representation = torch.cat((representationX, representationY), dim=1)
        # representation = representationX * representationY
        pred_task = self.fc_task(representation)
        output = {}
        representation_4mCF = representation[(type == 0).nonzero(as_tuple=True)[0]]
        if representation_4mCF.shape[0] != 0:
            output['0'] = self.classification_4mCF(representation_4mCF)

        representation_4mCS = representation[(type == 1).nonzero(as_tuple=True)[0]]
        if representation_4mCS.shape[0] != 0:
            output['1'] = self.classification_4mCS(representation_4mCS)

        representation_4mCC = representation[(type == 2).nonzero(as_tuple=True)[0]]
        if representation_4mCC.shape[0] != 0:
            output['2'] = self.classification_4mCC(representation_4mCC)

        representation_4mCT = representation[(type == 3).nonzero(as_tuple=True)[0]]
        if representation_4mCT.shape[0] != 0:
            output['3'] = self.classification_4mCT(representation_4mCT)

        representation_5hmCH = representation[(type == 4).nonzero(as_tuple=True)[0]]
        if representation_5hmCH.shape[0] != 0:
            output['4'] = self.classification_5hmCH(representation_5hmCH)

        representation_5hmCM = representation[(type == 5).nonzero(as_tuple=True)[0]]
        if representation_5hmCM.shape[0] != 0:
            output['5'] = self.classification_5hmCM(representation_5hmCM)

        representation_6mAA = representation[(type == 6).nonzero(as_tuple=True)[0]]
        if representation_6mAA.shape[0] != 0:
            output['6'] = self.classification_6mAA(representation_6mAA)

        representation_6mACEL = representation[(type == 7).nonzero(as_tuple=True)[0]]
        if representation_6mACEL.shape[0] != 0:
            output['7'] = self.classification_6mACEL(representation_6mACEL)

        representation_6mACEQ = representation[(type == 8).nonzero(as_tuple=True)[0]]
        if representation_6mACEQ.shape[0] != 0:
            output['8'] = self.classification_6mACEQ(representation_6mACEQ)

        representation_6mAD = representation[(type == 9).nonzero(as_tuple=True)[0]]
        if representation_6mAD.shape[0] != 0:
            output['9'] = self.classification_6mAD(representation_6mAD)

        representation_6mAF = representation[(type == 10).nonzero(as_tuple=True)[0]]
        if representation_6mAF.shape[0] != 0:
            output['10'] = self.classification_6mAF(representation_6mAF)

        representation_6mAH = representation[(type == 11).nonzero(as_tuple=True)[0]]
        if representation_6mAH.shape[0] != 0:
            output['11'] = self.classification_6mAH(representation_6mAH)

        representation_6mAR = representation[(type == 12).nonzero(as_tuple=True)[0]]
        if representation_6mAR.shape[0] != 0:
            output['12'] = self.classification_6mAR(representation_6mAR)

        representation_6mAS = representation[(type == 13).nonzero(as_tuple=True)[0]]
        if representation_6mAS.shape[0] != 0:
            output['13'] = self.classification_6mAS(representation_6mAS)

        representation_6mATT = representation[(type == 14).nonzero(as_tuple=True)[0]]
        if representation_6mATT.shape[0] != 0:
            output['14'] = self.classification_6mATT(representation_6mATT)

        representation_6mATO = representation[(type == 15).nonzero(as_tuple=True)[0]]
        if representation_6mATO.shape[0] != 0:
            output['15'] = self.classification_6mATO(representation_6mATO)

        representation_6mAX = representation[(type == 16).nonzero(as_tuple=True)[0]]
        if representation_6mAX.shape[0] != 0:
            output[16] = self.classification_6mAX(representation_6mAX)

        return output, representation, pred_task
