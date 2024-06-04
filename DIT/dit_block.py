import torch
import torch.nn as nn
import math


class DiTBlock(nn.Module):

    def __init__(self, emb_size, head_num):
        super().__init__()

        self.emb_size = emb_size
        self.head_num = head_num

        self.gamma1 = nn.Linear(emb_size, emb_size)
        self.beta1 = nn.Linear(emb_size, emb_size)
        self.alpha1 = nn.Linear(emb_size, emb_size)

        self.gamma2 = nn.Linear(emb_size, emb_size)
        self.beta2 = nn.Linear(emb_size, emb_size)
        self.alpha2 = nn.Linear(emb_size, emb_size)

        # layer norm
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

        # multi-head self-attention
        self.wq = nn.Linear(emb_size, head_num*emb_size)
        self.wk = nn.Linear(emb_size, head_num*emb_size)
        self.wv = nn.Linear(emb_size, head_num*emb_size)
        self.lv = nn.Linear(head_num*emb_size, emb_size)

        # feed forward
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size*4),
            nn.ReLU(),
            nn.Linear(emb_size*4, emb_size)
        )

    
    def forward(self, x, condition):

        # condition embedding
        gamma1 = self.gamma1(condition)
        beta1 = self.beta1(condition)
        alpha1 = self.alpha1(condition)
        gamma2 = self.gamma2(condition)
        beta2 = self.beta2(condition)
        alpha2 = self.alpha2(condition)
        
        # layer norm 
        y = self.ln1(x)

        # scale & shift
        y = y*(1+gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        
        # multi-head attention
        q = self.wq(y)      # (batch,seq_len,nhead*emb_size)
        k = self.wk(y)      # (batch,seq_len,nhead*emb_size)
        v = self.wv(y)      # (batch,seq_len,nhead*emb_size)

        q = q.view(q.size(0), q.size(1), self.head_num, self.emb_size).permute(0, 2, 1, 3)  # (batch,nhead,seq_len,emb_size)
        k = k.view(k.size(0), k.size(1), self.head_num, self.emb_size).permute(0, 2, 3, 1)  # (batch,nhead,emb_size,seq_len)
        v = v.view(v.size(0), v.size(1), self.head_num, self.emb_size).permute(0, 2, 1, 3)  # (batch,nhead,seq_len,emb_size)

        att = q@k/math.sqrt(q.size(2))  # (batch, num_head, seq_len, seq_len)
        att = torch.softmax(att, dim=-1)    # (batch,nhead,seq_len,seq_len)

        y = att@v

        y = y.permute(0, 2, 1, 3)   # (batch, seq_len, num_head, emb_size)

        y = y.reshape(y.size(0),y.size(1),y.size(2)*y.size(3))  # (batch,seq_len,nhead*emb_size)

        y=self.lv(y)    # (batch,seq_len,emb_size)

        # scale

        y = y*alpha1.unsqueeze(1)

        # residual connection
        y = x+y

        # layer norm
        z = self.ln2(y)

        # scale & shift
        z = z*(1+gamma2.unsqueeze(1)) + beta2.unsqueeze(1)

        # feed forward
        z = self.ff(z)
        
        # scale
        z = z * alpha2.unsqueeze(1)

        # residual connection
        out = z+y

        return out



if __name__=='__main__':
    dit_block=DiTBlock(emb_size=16, head_num=4)
    
    x=torch.rand((5,49,16))
    cond=torch.rand((5,16))
    
    outputs=dit_block(x,cond)
    print(outputs.shape)



        



