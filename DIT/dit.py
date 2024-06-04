from torch import nn 
import torch 
from time_emb import TimeEmbedding
from dit_block import DiTBlock
 

class DiT(nn.Module):
    def __init__(self, img_size, patch_size, in_channel, emb_size, label_num, dit_num, head):
        super().__init__()

        self.patch_size = patch_size
        self.patch_count=img_size//self.patch_size
        self.channel=in_channel
        
        # Patchify
        self.conv=nn.Conv2d(in_channels=in_channel,out_channels=in_channel*patch_size**2,kernel_size=patch_size,padding=0,stride=patch_size) 
        self.patch_emb=nn.Linear(in_features=in_channel*patch_size**2,out_features=emb_size) 
        self.patch_pos_emb=nn.Parameter(torch.rand(1,self.patch_count**2,emb_size))

        # Time embdding
        self.time_emb = nn.Sequential(
            TimeEmbedding(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

        # Label embedding (condition)
        self.label_embedding = nn.Embedding(num_embeddings=label_num, embedding_dim=emb_size)
        
        # Dit blocks
        self.dits = nn.ModuleList()
        for i in range(dit_num):
            self.dits.append(DiTBlock(emb_size, head))

        # layer norm
        self.ln = nn.LayerNorm(emb_size)

        # linear back to patch
        self.linear = nn.Linear(emb_size, in_channel*patch_size**2)

    def forward(self, x, t, y):

        # label emb
        y_emb = self.label_embedding(y)

        # time emb
        t_emb = self.time_emb(t)

        cond = y_emb + t_emb

        x = self.conv(x)
        x = x.permute(0,2,3,1)  # (bs, patch, patch, channels)
        x = x.view(x.size(0), self.patch_count*self.patch_count, x.size(3))     # (bs, patch*patch, channels)
        x = self.patch_emb(x)
        x += self.patch_pos_emb        # add positional embedding

        for dit in self.dits:
            x = dit(x, cond)

        x = self.ln(x)

        x = self.linear(x)

        x = x.view(x.size(0), self.patch_count, self.patch_count, self.channel, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 2, 4, 5) # bs, channel, patch_count, patch_count, ps, ps
        x = x.permute(0, 1, 2, 4, 3, 5) # bs, c, pc, ps, pc, ps
        x = x.reshape(x.size(0), self.channel, self.patch_count*self.patch_size, self.patch_count*self.patch_size) # bs, c, w, h
        
        return x
    
if __name__=='__main__':
    dit=DiT(img_size=28,patch_size=4,in_channel=1,emb_size=64,label_num=10,dit_num=3,head=4)
    x=torch.rand(5,1,28,28)
    t=torch.randint(0,1000,(5,))
    y=torch.randint(0,10,(5,))
    outputs=dit(x,t,y)
    print(outputs.shape)
