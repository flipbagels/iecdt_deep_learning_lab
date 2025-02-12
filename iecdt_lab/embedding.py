import torch 
import numpy as np


"""
https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
"""

def embed_2d_sincos_pos(embed_dim, grid_size, cls_token=False, extra_token=0):
    """
    grid_size: int of the grid heigh and width
    returns: position_emebedding of shape
    [grid_size * grid_size, embed_dim] or [1+grid_size * grid_size, embed_dim] if cls_token=True
    """

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h) #here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = embed_2d_sincos_pos_from_grid(embed_dim, grid)
    if cls_token and extra_token > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim], 
            dtype=np.float32), pos_embed], axis=0)
    return pos_embed

def embed_2d_sincos_pos_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    #use half of dim to encode grid_h 
    emb_h = embed_1d_sin_cos_from_grid(embed_dim // 2, grid[0]) #(H*W, D/2)
    emb_w = embed_1d_sin_cos_from_grid(embed_dim // 2, grid[1]) #(H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) #(H*W, D)
    return emb

def embed_1d_sin_cos_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim/2.
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1) #(M,)
    out = np.einsum('m,d->md', pos, omega) #(M, D/2), outer product

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1) #(M, D)

    return emb

def embed_1d_sincos_temp(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return embed_1d_sin_cos_from_grid(embed_dim, pos)