
# import torch.nn.functional as F
import torch.nn as nn
import torch

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x): # x.shape: (3*B, split_dim, N)
        x = self.conv1(x)
        x = self.pointwise_conv(x)
        return x

class Agg_0(nn.Module):
    def __init__(self, seg_dim):
        super().__init__()
        self.conv = SeparableConv2d(seg_dim*3, seg_dim, 3, 1, 1)
        self.norm = nn.LayerNorm(seg_dim)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        b, c, n = x.shape
        x = self.act(self.norm(x.reshape(b, c, -1).permute(0, 2, 1)))

        return x

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

    def forward(self, values, keys, query):#, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # n_, h_, q_, k_, d_ = queries.shape[0], queries.shape[2], queries.shape[1], keys.shape[1], queries.shape[-1]
        # energy = (queries.permute(0, 2, 1, 3).reshape(n_ * h_, q_, d_) @ keys.permute(0, 2, 3, 1).reshape(n_ * h_, d_,
        #         k_)).reshape(n_, h_, q_, k_)
        # print('energy.max', energy.max())
        # print('energy.min',energy.min())
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # # Mask padded indices so their weights become 0
        # if mask is not None:
        #     energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        # temp = energy / (self.embed_size ** (1 / 2))
        # attention = torch.softmax(temp, dim=3)
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be (N, query_len, embed_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        # self.num_heads = num_heads
        # self.head_dim = embed_dim // num_heads

        # self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim)
        # self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.attention = SelfAttention(embed_dim, num_heads)

    def forward(self, x):
        B, N, D = x.shape

        # Linearly transform input to Q, K, V
        x = x.reshape(3, B//3, N, D)#.permute(1, 2, 0, 3)
        qkv = x
        # qkv = x.reshape(B//3, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Split Q, K, V

        output = self.attention(v, k, q)

        # # Compute self-attention scores
        # attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        # attn_probs = F.softmax(attn_scores, dim=-1)
        #
        # # Apply attention to values
        # attn_output = torch.matmul(attn_probs, v)
        # attn_output = attn_output.permute(0, 2, 1, 3).reshape(B//3, N, D)
        #
        # # Linearly transform the output
        # output = self.out_linear(attn_output)

        return output

# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#
#         # self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim)
#         self.out_linear = nn.Linear(embed_dim, embed_dim)
#
#     def forward(self, x):
#         B, N, D = x.shape
#
#         # Linearly transform input to Q, K, V
#         x = x.reshape(3, B//3, N, D).permute(1, 2, 0, 3)
#         qkv = x.reshape(B//3, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # Split Q, K, V
#
#         # Compute self-attention scores
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
#         attn_probs = F.softmax(attn_scores, dim=-1)
#
#         # Apply attention to values
#         attn_output = torch.matmul(attn_probs, v)
#         attn_output = attn_output.permute(0, 2, 1, 3).reshape(B//3, N, D)
#
#         # Linearly transform the output
#         output = self.out_linear(attn_output)
#
#         return output

# x: the input token with shape of (B, N, D), B is batch size, N=H*W, D is dimension
# qkv_mapping(): linear mapping (in=D, out=D*3) to generate Q, K, V
# att(): efficient multi-head Q-K-V computation
# token_ensemble(): linear mapping (in=out=D) to combine the outputs from the attention and non-attentionbranches
# act: activation function, implemented by HardSwish
# norm: normalization function, implemented by LayerNorm
# The aggregator is implemented by a depth-wise convolution (channels=groups=D//5) following a linear mapping
class GMA_Block(nn.Module):
    def __init__(self):
        super(GMA_Block, self).__init__()
        self.norm = nn.LayerNorm((192,), eps=1e-12, elementwise_affine=True)#256
        self.act = nn.Hardswish()
        self.linear_mapping = nn.Linear(768, 768*3)
        self.multihead_attention = MultiHeadAttention(576, num_heads=8)#512
        self.token_ensemble = nn.Linear(768, 768)

        # self.aggregator_pre_att_3x3 = SeparableConv2d(256, 256, 2, 1, 0)
        # self.aggregator_pre_att_5x5 = SeparableConv2d(256, 256, 3, 1, 1)
        self.aggregator_pre_att_3x3 = SeparableConv2d(192, 192, 3, 1, 1)
        self.aggregator_pre_att_5x5 = SeparableConv2d(192, 192, 5, 1, 2)
        self.aggregator_pre_att_7x7 = SeparableConv2d(192, 192, 7, 1, 3)
        self.aggregator_non_att_3x3 =Agg_0(192)


    def forward(self, x):
        B, N, D = x.shape # B: 64    N: 41    D: 768
        split_dim = D // 4#5

        # Generate Q/K/V
        # x.shape: (B, N, D)-》(B, N, 3*D)-》(B, N, 3, D)-》(3, B, N, D)-》(3*B, N, D)
        qkv = self.linear_mapping(x).reshape(B, N, 3, D).permute(2, 0, 1, 3).reshape(3 * B, N, D)
        qkv = qkv.transpose(1, 2).view(3 * B, D, N)
        qkv = qkv.split([split_dim] * 4, dim=1) # (3*B, N, split_dim)    split_dim = D//4
        # Now qkv[i] is the i-th branch with shape of (3*B, split_dim, N)
        # (3*B, split_dim, N) -》(3*B, N, split_dim)
        # qkv_pre_att_0 = self.act(self.norm(qkv[0].permute(0,2,1))) # qkv_pre_att_0/1/2.shape: (3*B, N, split_dim)
        # Generate group proxies via different aggregators
        qkv_pre_att_0 = self.act(self.norm(self.aggregator_pre_att_3x3(qkv[0]).permute(0,2,1)))
        qkv_pre_att_1 = self.act(self.norm(self.aggregator_pre_att_5x5(qkv[1]).permute(0,2,1)))
        qkv_pre_att_2 = self.act(self.norm(self.aggregator_pre_att_7x7(qkv[2]).permute(0,2,1))) #torch.Size([240, 26, 256])  B/N/D
        # qkv_pre_att_3 = self.act(self.norm(aggregator_pre_att_7x7(qkv[3])))

        # Non-attention branch  (3*B, split_dim, N)-》(3, B,split_dim, N)-》(B, 3,split_dim, N)-> (B, 3 * split_dim, N)
        qkv_non_att = qkv[3].reshape(3, B,split_dim, N).permute(1, 0, 2, 3).reshape(B, 3 * split_dim, N)
        x_non_att = self.act(self.norm(self.aggregator_non_att_3x3(qkv_non_att))) # x_non_att.shape:(B, N, split_dim)

        # Efficient multi-head Q-K-V self-Attention. We ignore the number of heads for brevity
        # Its input is (3*B, D*4/5, H, W), output is (B, D*4/5, H, W)  qkv_input.shape: (3*B, N, 3*split_dim)
        qkv_input = torch.cat([qkv_pre_att_0, qkv_pre_att_1, qkv_pre_att_2], dim=2)#
        # x_att = att(qkv_input)
        # multihead_attention = MultiHeadAttention(576, num_heads=8)#D 576
        x_att = self.multihead_attention(qkv_input)

        # combine the outputs from attention and the non-attention branch
        x = torch.cat([x_att, x_non_att], dim=2)  # the shape becomes (B, D, H, W)
        # x = x.reshape(B, D, N).permute(0, 2, 1)  # the shape becomes (B, N, D)
        x = self.token_ensemble(x)
        return x


# x = torch.randn(80, 26, 768)
# GMA = GMA_Block()
# x = GMA(x)
#
# print("hello")