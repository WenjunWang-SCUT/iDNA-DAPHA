import torch
import torch.nn as nn
# import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        #2X498X20  ->  2X498X2X10
        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        # print('queries.max',queries.max())
        # print('queries.min',queries.min())
        # print('keys.max',keys.max())
        # print('keys.min',keys.min())

        # n_, h_, q_, k_, d_ = queries.shape[0], queries.shape[2], queries.shape[1], keys.shape[1], queries.shape[-1]
        # energy = (queries.permute(0, 2, 1, 3).reshape(n_ * h_, q_, d_) @ keys.permute(0, 2, 3, 1).reshape(n_ * h_, d_,
        #         k_)).reshape(n_, h_, q_, k_)
        # print('energy.max', energy.max())
        # print('energy.min',energy.min())
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # print('energy1.max', energy1.max())
        # print('energy1.min',energy1.min())
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        # temp = energy / (self.embed_size ** (1 / 2))
        # attention = torch.softmax(temp, dim=3)
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # self.word_embedding = nn.Embedding.from_pretrained(embedding_tsor) #nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # self.position_embedding = np.array([
        #     [pos / (10000**(2.0 * (j // 2) / embed_size)) for j in range(embed_size)]
        #     for pos in range(max_length)])
        # self.position_embedding[:, 0::2] = np.sin(self.position_embedding[:, 0::2])
        # self.position_embedding[:, 1::2] = np.cos(self.position_embedding[:, 1::2])
        # self.position_embedding = torch.from_numpy(self.position_embedding).cuda().double()

        # self.position_embedding = nn.Parameter(torch.FloatTensor(max_length, embed_size))

        self.layers = TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask): #x.shape:torch.Size([100, 498])
        N, seq_length, _ = x.shape #ÿ��batch��������N:100  seq_length:498 ÿ�����������е�ÿ��Ԫ�ؾ���һ������
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device) #positions.shape:torch.Size([1, 300]) [[  0,   1,   2,   ..., 297, 298, 299]]
        out = self.dropout(
			x + self.position_embedding(positions)
		)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        out = self.layers(out, out, out, mask)

        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        src_pad_idx,
        embed_size=768, #embed_size=512,  768,    20
        num_layers=1,
        forward_expansion=4,
        heads=8, #heads=8, 2
        dropout=0,#0
        device="cuda",
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.device = device
        # self.fc1 = nn.Linear(max_length*embed_size, 64)
        # self.out = nn.Linear(64, 1)
        # self.bn = nn.BatchNorm1d(64)
        #
        # self.sigmoid = nn.Sigmoid()
        # self.fctemp = nn.Sequential(
        #     # nn.BatchNorm1d(512),
        #     # nn.ReLU(),
        #
        #     nn.Dropout(p=0.9),
        #     nn.Linear(64, 1),  # 59904  60255
        #     nn.Sigmoid()
        # )

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) #src_mask.shape:torch.Size([1, 1, 1, 300]) [[[[ True,  True,  True,  ..., False, False, False]]]]
        # (N, 1, 1, src_len) src_mask�е�falseֵ����ʶ����λ��
        return src_mask.to(self.device)

    def forward(self, src, seqvalue_train):
        src_mask = self.make_src_mask(seqvalue_train)#self.make_src_mask(src) #src.shape:torch.Size([1, 300]) tensor([[ 2237,  7255,  6075,  ...,   0,     0,     0]], device='cuda:0')
        enc_src = self.encoder(src, src_mask)

        # # logit = torch.flatten(enc_src, start_dim=1)
        # x = enc_src.reshape(enc_src.shape[0], -1)
        # x = self.bn(self.fc1(x))
        # x = self.out(x)
        # x = self.sigmoid(x)
        return enc_src#, x.squeeze(1)
