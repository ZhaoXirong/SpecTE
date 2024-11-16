from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from positional_encodings.torch_encodings import PositionalEncoding1D
import torch.nn.functional as F

class SpecTE_Estimator(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, inpute_size=3450,num_lable = 3, patch_size=30,global_pool=False, **kwargs):
        super(SpecTE_Estimator, self).__init__(**kwargs)
        embed_dim=self.embed_dim
        self.num_lable = num_lable
        self.patch_embed = PatchEmbed(inpute_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches

        pos_enc_1d_model = PositionalEncoding1D(embed_dim)
        pos_embed =pos_enc_1d_model(torch.zeros(1, num_patches + 1, embed_dim)) # 位置编码  固定sin-cos嵌入  fixed sin-cos embedding
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
        
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm
            
          
        self.mu = nn.Linear(in_features=embed_dim, out_features=num_lable,  )

        self.sigma = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=num_lable),
            nn.Softplus()  # 保证输出为正数
            )

    def forward(self, x):
        B = x.shape[0]

        x,_ = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]

        outcome = self.head_drop(x)


        return self.mu(outcome), self.sigma(outcome) 
                

class PatchEmbed(nn.Module):
    """
    1D Flux to Patch Embedding
    """
    def __init__(self, flux_size: int = 224,
            patch_size: int = 32,
            embed_dim: int = 32,
            norm_layer=None):
        super().__init__()
        self.flux_size = flux_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=int(patch_size), stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.num_patches = flux_size // patch_size + (1 if flux_size % patch_size else 0)
        
        
    def forward(self, x):
        # x [b,3456]
        _, flux_dim= x.shape  

        # padding
        # 如果输入flux的dim不是patch_size的整数倍，需要进行padding
        pad_input = flux_dim % self.patch_size != 0
        if pad_input: 
            x = F.pad(x, (0, self.patch_size - flux_dim % self.patch_size))
        # 下采样patch_size倍
        x = x.unsqueeze(1)
        x = self.proj(x)    # 宽高缩小patch_size(4)倍 [batch,embed_dim,432]
        _, _,  W = x.shape  
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.transpose(1, 2) # [2,432,16]
        x = self.norm(x)
        return x, W



if __name__ == '__main__':
    

    # vision_transformer = ViT(
    #     image_size=256,patch_size=32,
    #     num_classes=4,dim=1024,
    #     depth=1,heads=2,mlp_dim=2048
    #     )
    net = SpecTE_Estimator(inpute_size=3456, num_lable = 3, patch_size=16, 
                 embed_dim=16, depth=4, num_heads=8,
                 mlp_ratio=4., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    
    # .cuda()
    x = torch.zeros(10,3456)
    # x = torch.zeros(1,3, 224, 224).cuda()
    mu,sigma = net(x)

    print("mu:",mu)
    print("mu.shape:",mu.shape)
    print("sigma:",sigma)
    print("sigma.shape:",sigma.shape)




