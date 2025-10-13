import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle

# from transformers import BertTokenizer, BertModel, BertConfig
from transformers_local.src.transformers.models.bert.tokenization_bert import BertTokenizer
from transformers_local.src.transformers.models.bert.configuration_bert import BertConfig
from transformers_local.src.transformers.models.bert.modeling_bert import BertModel

MER = 3

SEQUENCE = ["CTCGAACGGCGTCCCGAACTCGACGACGGCGCGCGAGCGGA"]
# SEQUENCE = ["GTATAGCGTGCCAGTGTGCTCGCCATGGACTGCGGCGGTTA"]
# SEQUENCE = ["GAAGCAGAGTCCCTGGGAGGCGCCACTGGTCAGGCCTGGAC"]
# SEQUENCE = ["ACTTCCAAGAGTGATAATTGAATGAACCTAAAATCACCATA"]


'''DNA bert model'''
class DNABERT(nn.Module):
    def __init__(self, config):
        super(DNABERT,self).__init__()
        self.config = config

        # 加载预训练模型参数
        self.kmer = config.kmer
        if self.kmer == 3:
            self.pretrainpath = '../pretrain/DNAbert_3mer'
        elif self.kmer == 4:
            self.pretrainpath = '../pretrain/DNAbert_4mer'
        elif self.kmer == 5:
            self.pretrainpath = '../pretrain/DNAbert_5mer'
        elif self.kmer == 6:
            self.pretrainpath = '../pretrain/DNAbert_6mer'

        self.setting = BertConfig.from_pretrained(
            self.pretrainpath,
            num_labels=2,
            finetuning_task="dnaprom",
            cache_dir=None,
            output_attentions=True
        )

        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainpath)
        self.bert = BertModel.from_pretrained(self.pretrainpath, config=self.setting)

    def forward(self, seqs):
        seqs = list(seqs)
        kmer = [[seqs[i][x:x + self.kmer] for x in range(len(seqs[i]) + 1 - self.kmer)] for i in range(len(seqs))]
        kmers = [" ".join(kmer[i]) for i in range(len(kmer))]
        token_seq = self.tokenizer(kmers, return_tensors='pt')
        input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq[
            'attention_mask']
        if self.config.cuda:
            representation = self.bert(input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda())
        else:
            representation = self.bert(input_ids, token_type_ids, attention_mask)

        return representation

class FusionBERT(nn.Module):
    def __init__(self, config):
        super(FusionBERT,self).__init__()
        self.config = config

        self.config.kmer = self.config.kmers[0]
        self.bertone = DNABERT(self.config)

        self.config.kmer = self.config.kmers[1]
        self.berttwo = DNABERT(self.config)

        # self.Ws = torch.randn(1, 768).cuda()
        # self.Wh = torch.randn(1, 768).cuda()

        self.classification = nn.Sequential(
            nn.Linear(768*2, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, seqs):
        representationX = self.bertone(seqs)
        representationY = self.berttwo(seqs)
        # F = torch.sigmoid(self.Ws * representationX["pooler_output"] + self.Wh * representationX["pooler_output"])
        F = torch.sigmoid(torch.cat((representationX["pooler_output"], representationY["pooler_output"]), dim=1))
        return F, representationX, representationY


def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def get_attention(representation, start, end):
    attention = representation[-1]
    # print(attention)
    # print(len(attention))
    # print(attention[0].shape)
    # print(attention[0].squeeze(0).shape) torch.Size([12, 43, 43])

    """
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    attn = format_attention(attention)
    # print(attn.shape) torch.Size([12, 12, 43, 43])

    attn_score = []
    for i in range(1, attn.shape[3] - 1):
        # only use cls token, because use pool out
        attn_score.append(float(attn[start:end + 1, :, 0, i].sum()))

    # print(len(attn_score)) 41
    return attn_score


def get_real_score(attention_scores, kmer, metric):
    # make kmers project to real sequence
    counts = np.zeros([len(attention_scores) + kmer - 1])
    real_scores = np.zeros([len(attention_scores) + kmer - 1])

    if metric == "mean":
        for i, score in enumerate(attention_scores):
            for j in range(kmer):
                # count the number of one nucleotide
                counts[i + j] += 1.0
                real_scores[i + j] += score

        real_scores = real_scores / counts
    else:
        pass
    real_scores = (real_scores - real_scores.min()) / (real_scores.max() - real_scores.min())
    return real_scores


def load_params(model, param_path):
    pretrained_dict = torch.load(param_path)
    new_model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    model.load_state_dict(new_model_dict)


def visualize_one_sequence(config):
    model = FusionBERT(config)
    load_params(model, config.path_params)
    model = model.cuda()
    F, representationX, representationY = model(SEQUENCE)

    # print(get_attention(representationX, 11, 11))
    if MER == 3:
        attention = get_attention(representationX, 11, 11)
    else:
        attention = get_attention(representationY, 11, 11)

    attention_scores = np.array(attention).reshape(np.array(attention).shape[0], 1)

    if MER == 3:
        real_scores = get_real_score(attention_scores, 3, "mean")
    else:
        real_scores = get_real_score(attention_scores, 6, "mean")

    scores = real_scores.reshape(1, real_scores.shape[0])

    # ave is not used in the end
    ave = np.sum(scores) / scores.shape[1]
    # print(ave)
    # print(scores)

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    # plot
    sns.set()
    # scores = (scores - scores.min()) / ( scores.max() - scores.min())
    # ax = sns.heatmap(scores, cmap='Greens', vmin=0, vmax=1)
    ax = sns.heatmap(scores, cmap='YlGnBu', vmin=0, vmax=1)
    plt.show()


def interface(configpath, configmodel, seqs, type):
    config = pickle.load(open(configpath, 'rb'))
    config.path_params = configmodel
    model = FusionBERT(config)
    load_params(model, config.path_params)
    model = model.cuda()

    from tqdm import tqdm

    pos_atten_scores = []
    pbar = tqdm(seqs)
    for seq in pbar:
        # print(seq)
        F, representationX, representationY = model([seq])

        attentionX = get_attention(representationX, 11, 11)
        attentionY = get_attention(representationY, 11, 11)

        real_scores_X = get_real_score(attentionX, 3, "mean")
        real_scores_Y = get_real_score(attentionY, 6, "mean")

        scores = []
        for i in range(41):
            if type == 3:
                scores.append(real_scores_X[i])
            elif type == 6:
                scores.append(real_scores_Y[i])

        pos_atten_scores.append(scores)

    return pos_atten_scores



if __name__ == "__main__":
    # dataset_name = '6mAS'
    #
    # configPath = "./result/" + dataset_name + "/config.pkl"
    # paramsPath = "./result/" + dataset_name + "/BERT" + r", ACC[0.828].pt"
    configPath = "../result/trainCross/8e-06_160_4mC_Tolypocladiumnoacalpre5/config.pkl"
    paramsPath = "../result/trainCross/8e-06_160_4mC_Tolypocladiumnoacalpre5/ACC4_0.74481273652616.pt"
    #paramsPath = "../result/trainCross/8e-06_160_4mC_Tolypocladiumnoacalpre5" + "/BERT" + r", ACC[0.85363383134970883].pt"

    config = pickle.load(open(configPath, 'rb'))
    config.path_params = paramsPath

    os.environ["CUDA_VISIBLE_DEVICES"] = '7'

    visualize_one_sequence(config)
