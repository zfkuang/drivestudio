from typing import Tuple
import torch
import torch.nn as nn
from promptda.attention import LayerScale, MemEffAttention, Mlp
from promptda.utils.pe_utils import PositionEmbeddingRandom
from torch import Tensor

class GeneralPromptModel(nn.Module):
    def __init__(self, prompt_stage=[3], **kwargs):
        super().__init__()
        self.prompt_stage = prompt_stage
        self.prompt_idmap = {stage: idx for idx, stage in enumerate(self.prompt_stage)}
        self.prompt_model = nn.ModuleList([SelfAttnPromptModel() for _ in range(len(self.prompt_stage))])

    def forward(self, features, prompt_depth, prompt_mask, patch_h, patch_w):
        for i in range(len(features)):
            if i in self.prompt_stage:
                features[i][0] = self.prompt_model[self.prompt_idmap[i]](
                    features[i][0],
                    prompt_depth,
                    prompt_mask,
                    patch_h,
                    patch_w,
                )
        return features


class SelfAttnPromptModel(nn.Module):
    def __init__(
        self,
        transformer_dim=1024,
        num_blocks=4,
        num_heads=4,
        pe="qk",
        image_pe_method="patch",
        **kwargs,
    ):
        """
        Predicts masks given an image and prompt embeddings using a transformer architecture.
        
        Args:
            transformer_dim: Channel dimension of the transformer
            num_blocks: Number of transformer blocks
            num_heads: Number of attention heads
            pe: Positional encoding type ("qk", "apply", or "normal")
            image_pe_method: Method for image positional encoding ("patch" or "image")
        """
        super().__init__()
        self.pe = pe
        pe_dim = transformer_dim // 2
        if self.pe == "apply":
            pe_dim = pe_dim // num_heads
            
        self.pe_layer = PositionEmbeddingRandom(pe_dim, image_pe_method=image_pe_method)
        self.prompt_blocks = nn.ModuleList([
            SelfAttenPromptBlock(
                dim=transformer_dim,
                num_heads=num_heads,
                first_block=(i == 0),
                pe=pe
            ) for i in range(num_blocks)
        ])
        
        self.depth2feature = nn.Sequential(
            nn.Linear(1, transformer_dim // 2),
            nn.GELU(),
            nn.Linear(transformer_dim // 2, transformer_dim),
        )

    def forward(
        self,
        image_embeddings: Tensor,
        prompt_depth: Tensor,
        prompt_mask: Tensor,
        patch_h: int,
        patch_w: int,
    ) -> Tensor:
        """
        Process image embeddings with prompt guidance.
        
        Args:
            image_embeddings: Embeddings from the image encoder
            prompt_depth: Depth values for prompt points
            prompt_mask: Mask indicating valid prompt points
            patch_h: Height of feature patches
            patch_w: Width of feature patches
            
        Returns:
            Updated image embeddings
        """
        B, _, H, W = prompt_depth.shape
        image_pe = self.pe_layer((patch_h, patch_w)).permute(1, 2, 0)  # CxHxW -> HxWxC
        image_embeddings_list = []
        
        for b in range(B):
            valid_pts_num = (prompt_mask[b, 0] > 0.0).sum()
            
            # Skip processing if no valid prompt points
            if valid_pts_num == 0:
                image_embeddings_list.append(image_embeddings[b:(b+1)])
                continue
                
            # Extract valid prompt positions and depths
            sparse_depth_pos = (prompt_mask[b, 0] > 0.0).nonzero().float()
            sparse_depth_pos[:, 0] = (sparse_depth_pos[:, 0] + 0.5) / H
            sparse_depth_pos[:, 1] = (sparse_depth_pos[:, 1] + 0.5) / W
            sparse_depth = prompt_depth[b, 0][prompt_mask[b, 0] > 0.0]
            
            # Generate prompt embeddings and positional encodings
            prompt_embeddings = self.depth2feature(sparse_depth[:, None])[None, ...]  # 1, N, C
            prompt_pe = self.pe_layer._pe_encoding(sparse_depth_pos[None, :, [1, 0]])  # 1, N, C
            query_pe = image_pe.reshape(1, -1, image_pe.shape[-1])
            
            # Initialize query and prompt
            prompt = prompt_embeddings
            query = image_embeddings[b:(b+1)]
            
            # Process through prompt blocks
            for block in self.prompt_blocks:
                query, prompt = block(query, query_pe, prompt, prompt_pe)
                
            image_embeddings_list.append(query[..., :image_embeddings.shape[-1]])
            
        return torch.cat(image_embeddings_list, dim=0)


class SelfAttenPromptBlock(nn.Module):
    """
    Self-attention block for prompt-based processing that handles both query and context features.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        init_values: float = 0.0,
        first_block: bool = False,
        pe: str = "normal",
        **kwargs,
    ):
        super().__init__()
        self.first_block = first_block
        self.pe = pe

        # Attention components
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MemEffAttention(dim, num_heads=num_heads, pe=pe)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        # MLP components
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=dim * 4)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        # Separator tokens
        self.sep = nn.Parameter(torch.randn(1, 1, dim))
        if self.pe != "normal":
            self.sep_pe = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x: Tensor, x_pe: Tensor, context: Tensor, context_pe: Tensor) -> Tuple[Tensor, Tensor]:
        # Apply positional encoding for first block with normal PE
        if self.pe == "normal" and self.first_block:
            x = x + x_pe
            context = context + context_pe

        # Record original sequence lengths
        x_len, context_len = x.shape[1], context.shape[1]

        # Concatenate query, separator token, and context
        x = torch.cat([x, self.sep, context], dim=1)

        # Handle positional encoding
        if self.pe != "normal":
            x_pe = torch.cat([x_pe, self.sep_pe, context_pe], dim=1)
            x = x + self.ls1(self.attn(self.norm1(x), x_pe))
        else:
            x = x + self.ls1(self.attn(self.norm1(x)))

        # Apply MLP
        x = x + self.ls2(self.mlp(self.norm2(x)))

        # Split back into query and context
        query = x[:, :x_len, :]
        context = x[:, x_len + 1:x_len + 1 + context_len, :]

        return query, context
