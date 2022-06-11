import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# MLP
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# self-attention
class Attention(nn.Module):
    # 自注意力的实现，核心在于qkv后一个softmax+dropout
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        '''
        x.shape:(bs, num_patchs + 1, 1024)
        ori q k v.shape:(bs, num_patchs+1, 1024)
        q,k,v.shape:(bs, heads, num_patchs + 1, 1024//heads)
        
        这里使用的是matmul来进行矩阵的乘积
        Z = Q×K^T / √ dim
        Z = softmax(Z),最后一层
        Z = dropout(Z)
        out = Z×V
        '''
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # 原版的ViT的深度（depth）为6，也就是自注意力和前向网络（mlp）重复6次，
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        # 在Vision Transformer中，一共由6层该结构的模块
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# Vision Transformer
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = (image_size,image_size)    # eg. 32
        patch_height, patch_width = (patch_size,patch_size)    # eg. 32

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))       # 1 1 1024
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        '''
        :param img:图像
        :return: prob of class 类别概率
        '''
        '''第一步，将图像分为多个patch
        在这里，shape=(bs,c,8*32,8*32) --> (bs,8*8,32*32*3)
        然后，将图像降维，shape=(bs,8*8,32*32*3) --> (bs,8*8,1024)
        '''
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape       # bs 64 1024

        '''第二步，引入cls_tokens和Positioal Encoding
        1.ViT提出可学习的嵌入向量 Class Token，假设输入为3×3的图像块，向量数量N=9，它与9个向量一起输入到 Transformer 结构中，输出为10个编码向量
        然后用这个 Class Token 进行分类预测，Class Token 的作用就是寻找其他9个输入向量对应的类别。
        
        2.Positioal Encoding即位置编码，ViT 中的位置编码没有采用原版 Transformer 中的 sincos 编码，而是直接设置为可学习的 Positional Encoding
        在Transformer中，PE的引入是为了让模型知道每个字在句子中的顺序，也就是说，PE确定了每个表征向量的位置，Transformer中的PE使用的是sincos编码
        '''
        # cls_tokens嵌入向量
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        # cls_tokens = torch.repeat_interleave(self.cls_token,repeats=b,dim=0)  # 方式2
        x = torch.cat((cls_tokens, x), dim=1)   # 对其进行输入
        # pos_embedding位置编码(可直接学习的tensor)，不是cat操作，而是直接相加
        # tmp = self.pos_embedding[:, :(n + 1)]   # 1 65 1024
        # x += tmp                                # bs 65 1024
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        '''第三步，送入Transformer中'''
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

if __name__ == '__main__':
    x = torch.rand(2,3,256,256)
    model = ViT(image_size = 256,
                patch_size = 32,
                num_classes = 10,
                dim = 1024,
                depth = 6,
                heads = 16,
                mlp_dim = 2048,
                pool = 'cls', channels = 3, dim_head = 64,
                dropout = 0.1, emb_dropout = 0.1)
    y = model(x)
    print(y)