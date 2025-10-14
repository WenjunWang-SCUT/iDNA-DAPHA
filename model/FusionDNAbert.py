import torch
import torch.nn as nn
# import json
# import torch.nn.functional as F
# from pykalman import KalmanFilter

from model import DNAbert
# from model.pytorch_revgrad.module import RevGrad
# from .gmam import GMA_Block

'''bert Fusion 模型'''
class FusionBERT(nn.Module):
    def __init__(self, config):
        super(FusionBERT, self).__init__()
        self.config = config

        self.config.kmer = self.config.kmers[0]
        self.bertone = DNAbert.BERT(self.config)

        self.config.kmer = self.config.kmers[1]
        self.berttwo = DNAbert.BERT(self.config)
        
        # self.transformer = TransformerF.Transformer(
        #         src_vocab_size=64,
        #         src_pad_idx=0,
        #         max_length=41
        #         )#0.1
        # with open("../model/config.json", encoding='utf-8', errors='ignore') as json_data:
        #     config = json.load(json_data, strict=False)
        # att_DNA_args_3mer = {'n_nts': config['MODEL']['embedding_dim'], 'n_bins': 39,
        #                      'bin_rnn_size': config['MODEL']['hidden_dim'],
        #                      'num_layers': config['MODEL']['hidden_layers'],
        #                      'dropout': 0.1, 'bidirectional': True}
        # self.att_DNA_3mer = Lstm_atta_models.att_DNA(att_DNA_args_3mer, config['MODEL']['output_dim'])
        #
        # att_DNA_args_6mer = {'n_nts': config['MODEL']['embedding_dim'], 'n_bins': 36,
        #                 'bin_rnn_size': config['MODEL']['hidden_dim'], 'num_layers': config['MODEL']['hidden_layers'],
        #                 'dropout': 0.1, 'bidirectional': True}
        # self.att_DNA_6mer = Lstm_atta_models.att_DNA(att_DNA_args_6mer, config['MODEL']['output_dim'])

        # self.lstm = nn.LSTM(input_size = 768, hidden_size = 256, num_layers = 2,
        #                     batch_first=True, bidirectional=True)
        # self.attention = nn.MultiheadAttention(embed_dim = 256*2, num_heads = 8)
        # self.kalman_filter = KalmanFilter(n_dim_obs=512, n_dim_state=512)

        self.classification = nn.Sequential(
            nn.Linear(768 * 2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

        # for param in self.bertone.parameters():
        #     param.requires_grad = False
        # for param in self.berttwo.parameters():
        #     param.requires_grad = False

    def forward(self, seqs):#, seqvalue_feat
        # print(seqs)
        representationX = self.bertone(seqs)
        representationY = self.berttwo(seqs)


        representation = torch.cat((representationX, representationY), dim=1)
        # representation = representationX + representationY

        output = self.classification(representation)


        return output, representation
