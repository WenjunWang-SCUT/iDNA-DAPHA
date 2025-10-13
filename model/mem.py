import torch
import torch.nn as nn
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        
        self.d = 1536 #args.bottleneck_size 768  1536
        self.n = 64 #args.num_mem 64
        self.T = 16 #args.temper 16
        
        self.feats = nn.Parameter(torch.randn((self.n, self.d), requires_grad=True))
        self.w_k = nn.Parameter(torch.randn((self.d, self.d), requires_grad=True))
        self.w_q = nn.Parameter(torch.randn((self.d, self.d), requires_grad=True))
        
    def forward(self, x):
        """_summary_

        Args:
            x: img_feat, shape [bs, d]
        """
        key = torch.mm(self.feats, self.w_k) # [n, d]
        query = torch.mm(x, self.w_q) # [bs, d]
        value = self.feats
        
        key = F.normalize(key, dim=1).permute(1, 0).contiguous() # [d, n]
        query = F.normalize(query, dim=1) # [bs, d]
        
        sim = torch.mm(query, key) # [bs, n]
        atten = torch.softmax(sim * self.T, dim=1) # [bs, n]
        
        atten_feat = torch.mm(atten, value)
        
        return atten_feat