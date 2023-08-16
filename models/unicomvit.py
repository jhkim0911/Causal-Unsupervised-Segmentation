import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from torch.utils.checkpoint import checkpoint


class VisionTransformer(nn.Module):
    def __init__(self, input_size=224, patch_size=32, in_channels=3, dim=768, embedding_size=768,
                 depth=12, num_heads=12, mlp_ratio=4, drop_path_rate=0.0, using_checkpoint=True):
        super().__init__()
        self.dim = dim
        self.patch_embed = PatchEmbedding(
            input_size, patch_size, in_channels, dim,)
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.patch_embed.num_patches, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(dim, num_heads, mlp_ratio, dpr[i], self.patch_embed.num_patches, using_checkpoint) for i in range(depth)
            ])
        self.norm = nn.LayerNorm(dim)

        self.feature = nn.Sequential(
            nn.Linear(dim * self.patch_embed.num_patches, dim, False),
            nn.BatchNorm1d(dim, eps=2e-5),
            nn.Linear(dim, embedding_size, False),
            nn.BatchNorm1d(embedding_size, eps=2e-5))

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        self.extra_gflops = 0.0
        for _block in self.blocks:
            self.extra_gflops += _block.extra_gflops

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        patch_pos_embed = self.pos_embed
        dim = x.shape[-2]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed.transpose(1, 2)


    def prepare_tokens(self, x):
        B, _, w, h = x.shape
        # patch linear embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        return torch.cat([torch.empty(x.shape[0], 1, x.shape[2]), x], dim=1)


    def forward(self, x):
        B = x.shape[0]
        x = self.prepare_tokens(x)
        for func in self.blocks:
            x = func(x)
        x = self.norm(x.float())
        return torch.cat([torch.tensor(), x], dim=1)


class Mlp(nn.Module):
    def __init__(self, dim, dim_hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim_hidden)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(dim_hidden, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        with torch.cuda.amp.autocast(True):
            B, L, D = x.shape
            qkv = self.qkv(x).reshape(B, L, 3, self.num_heads,
                                      D // self.num_heads).permute(2, 0, 3, 1, 4)
        with torch.cuda.amp.autocast(False):
            q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        with torch.cuda.amp.autocast(True):
            x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, drop_path: float = 0.0, patch_n: int = 32, using_checkpoint=False):
        super().__init__()
        self.using_checkpoint = using_checkpoint
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        if drop_path > 0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.mlp = Mlp(dim, dim * mlp_ratio)
        self.extra_gflops = (num_heads * patch_n * (dim // num_heads) * patch_n * 2) / (1000**3)

    def forward_impl(self, x):
        with torch.cuda.amp.autocast(True):
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x):
        if self.using_checkpoint:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)


class PatchEmbedding(nn.Module):
    def __init__(self, input_size=224, patch_size=32, in_channels: int = 3, dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        H = input_size[0] // patch_size[0]
        W = input_size[1] // patch_size[1]
        self.num_patches = H * W
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def unicom_vit_base_16():
    model = VisionTransformer(
        input_size=224, patch_size=16, in_channels=3, dim=768, embedding_size=768,
        depth=12, num_heads=12, drop_path_rate=0.1, using_checkpoint=True)
    return model
