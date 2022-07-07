from vit_pytorch import ViT
import torch
import torchstat

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# img = torch.randn(1, 3, 256, 256)

# preds = v(img) # (1, 1000)

# torchstat.stat(v,(3,256,256))
from torchsummary import summary
summary(v,input_size=(3,256,256))