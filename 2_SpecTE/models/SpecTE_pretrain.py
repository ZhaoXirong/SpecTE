from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Block #PatchEmbed, 

from positional_encodings.torch_encodings import PositionalEncoding1D, Summer #位置编码

class SpecTE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    
    """
    def __init__(self, inpute_size=3450, patch_size=30,
                 embed_dim=32, depth=8, num_heads=8,
                 decoder_embed_dim=16, decoder_depth=2, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,drop_rate=0.):
        super().__init__()
        """
        初始化SpecTE模型。

        参数：
        inpute_size (int): 输入光谱的大小，默认为3450。
        patch_size (int): 将输入图像分割成补丁的大小，默认为30。
        embed_dim (int): 嵌入维度，默认为32。
        depth (int): Encoder的层数，默认为8。
        num_heads (int): Transformer注意力头的数量，默认为8。
        decoder_embed_dim (int): 解码器的嵌入维度，默认为512。
        decoder_depth (int): 解码器的层数，默认为8。
        decoder_num_heads (int): 解码器的注意力头数量，默认为16。
        mlp_ratio (float): MLP中隐层与嵌入维度的比例，默认为4。
        norm_layer (nn.Module): 归一化层，默认为nn.LayerNorm。
        norm_pix_loss (bool): 是否使用像素归一化损失，默认为False。
        """
        self.patch_size = patch_size
        # --------------------------------------------------------------------------
        # SpecTE encoder specifics  编码器部分
        self.patch_embed = PatchEmbed(inpute_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))   # class tocken
        p_enc_1d_model = PositionalEncoding1D(embed_dim)
        pos_embed =p_enc_1d_model(torch.zeros(1, num_patches + 1, embed_dim)) # 位置编码  固定sin-cos嵌入  fixed sin-cos embedding
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        self.drop = nn.Dropout(drop_rate)
        # --------------------------------------------------------------------------
        # SpecTE decoder specifics 解码器部分
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        decoder_pos_embed =p_enc_1d_model(torch.zeros(1, num_patches + 1, decoder_embed_dim)) # 位置编码  固定sin-cos嵌入  fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(decoder_pos_embed, requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # 变量初始化
        # initialization

        # 将patch_embed初始化为类似nn.Linear（而不是nn.Conv2d）
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))  #Xavier Uniform初始化的目的是在训练开始时保持每一层的激活值的尺度大致相同，从而避免梯度消失或爆炸的问题。

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
         # 初始化class token 正态分布
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        #初始化模型权重值。
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:（均匀分布）
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def flux_pad(self, flux):
        """
        FLUX裁剪成多个patch
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        
        _, flux_dim= flux.shape  

        pad_input = flux_dim % self.patch_size != 0
        if pad_input: 
            flux = F.pad(flux, (0, self.patch_size - flux_dim % self.patch_size))

        return flux




    def forward_encoder(self, x):
        # embed patches
        x,_ = self.patch_embed(x)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        x = rearrange(x, 'b h x -> b (h x)')
        return x

    def forward_loss(self, flux, pred):
        """
        flux: [N, length]
        pred: [N, L, patch]
        """

        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5

        target = self.flux_pad(flux)  

        loss = (pred - target) ** 2
        loss = loss.mean() # [N, L], mean loss per patch
        return loss

    def forward(self, flux, flux_high):
        latent = self.forward_encoder(flux)
        latent=self.drop(latent)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss = self.forward_loss(flux_high, pred)
        return loss, pred


 
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
    net = SpecTE( inpute_size=3450, patch_size=16, 
                 embed_dim=16, depth=4, num_heads=8,
                 decoder_embed_dim=16, decoder_depth=2, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False)

    # .cuda()
    x = torch.zeros(5,3450)
    # x = torch.zeros(1,3, 224, 224).cuda()
    loss, pred = net(x,x)


    print("loss:",loss)
    print("pred:",pred)
    print("pred.shape:",pred.shape)


    