# -*- coding: utf-8 -*-

# ----------------------------
# Import necessary libraries
# ----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 2-3 Input Layer
# ----------------------------
print("=======2-3 Input Layer=======")

class VitInputLayer(nn.Module): 
    def __init__(self, in_channels:int=3, emb_dim:int=384, num_patch_row:int=2, image_size:int=32):
        """
        Arguments:
            in_channels: Number of channels in the input image
            emb_dim: Length of the embedded vector
            num_patch_row: Number of patches in height direction. Defaults to 2 as in a 2x2 example
            image_size: Size of one side of the input image. It is assumed that the height and width of the input image are the same
        """
        super(VitInputLayer, self).__init__() 
        self.in_channels = in_channels 
        self.emb_dim = emb_dim 
        self.num_patch_row = num_patch_row 
        self.image_size = image_size
        
        # Number of patches
        ## For example, if dividing the input image into 2x2 patches, num_patch is 4 
        self.num_patch = self.num_patch_row**2

        # Patch size
        ## For example, if one side of the input image is 32, patch_size is 16 
        self.patch_size = int(self.image_size // self.num_patch_row)

        # Layer to split the input image into patches & embed patches all at once 
        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=self.emb_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )

        # Class token 
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, emb_dim) 
        )

        # Positional embedding
        ## As the class token is concatenated at the beginning,
        ## (num_patch + 1) positional embedding vectors of length emb_dim are prepared 
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_patch+1, emb_dim) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Input image. Shape: (B, C, H, W). [Equation (1)]
                B: Batch size, C: Channel count, H: Height, W: Width

        Returns:
            z_0: Input to ViT. Shape: (B, N, D).
                B: Batch size, N: Number of tokens, D: Length of the embedded vector
        """
        # Embedding & flattening patches [Equation (3)]
        ## Embedding patches (B, C, H, W) -> (B, D, H/P, W/P) 
        ## Here, P is the size of one side of the patch
        z_0 = self.patch_emb_layer(x)

        ## Flattening patches (B, D, H/P, W/P) -> (B, D, Np) 
        ## Here, Np is the number of patches (=H*W/Pˆ2)
        z_0 = z_0.flatten(2)

        ## Transpose axes (B, D, Np) -> (B, Np, D) 
        z_0 = z_0.transpose(1, 2)

        # Concatenating class token at the beginning of the patch embeddings [Equation (4)] 
        ## (B, Np, D) -> (B, N, D)
        ## Note that N = (Np + 1)
        ## Also, since cls_token has shape (1,1,D),
        ## use the repeat method to transform it to (B,1,D) before concatenating with the patch embeddings 
        z_0 = torch.cat(
            [self.cls_token.repeat(repeats=(x.size(0),1,1)), z_0], dim=1)

        # Adding positional embedding [Equation (5)] 
        ## (B, N, D) -> (B, N, D) 
        z_0 = z_0 + self.pos_emb
        return z_0

batch_size, channel, height, width = 2, 3, 32, 32
x = torch.randn(batch_size, channel, height, width) 
input_layer = VitInputLayer(num_patch_row=2) 
z_0 = input_layer(x)

# Check that it is (2, 5, 384)(=(B, N, D)). 
print(z_0.shape)



# ----------------------------
# 2-4 Self-Attention
# ----------------------------
print("=======2-4 Self-Attention=======")

class MultiHeadSelfAttention(nn.Module): 
    def __init__(self, emb_dim:int=384, head:int=3, dropout:float=0.):
        """ 
        Arguments:
            emb_dim: Length of the embedded vector
            head: Number of heads
            dropout: Dropout rate
        """
        super(MultiHeadSelfAttention, self).__init__() 
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim**0.5  # Square root of D_h. Factor for dividing qk^T

        # Linear layers for embedding inputs into q, k, v. [Equation (6)] 
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        # Not in Equation (7) but dropout layers are used in implementation 
        self.attn_drop = nn.Dropout(dropout)

        # Linear layer to embed the output of MHSA. [Equation (10)]
        ## Not in Equation (10) but dropout layers are used in implementation 
        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout) 
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """ 
        Arguments:
            z: Input to MHSA. Shape: (B, N, D).
                B: Batch size, N: Number of tokens, D: Length of the vector

        Returns:
            out: Output of MHSA. Shape: (B, N, D). [Equation (10)]
                B: Batch size, N: Number of tokens, D: Length of the embedded vector
        """

        batch_size, num_patch, _ = z.size()

        # Embedding [Equation (6)]
        ## (B, N, D) -> (B, N, D)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # Splitting q, k, v into heads [Equation (10)]
        ## First, split the vector into the number of heads (h)
        ## (B, N, D) -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        ## Rearrange for Self-Attention to work
        ## (B, N, h, D//h) -> (B, h, N, D//h)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # Dot product [Equation (7)]
        ## (B, h, N, D//h) -> (B, h, D//h, N)
        k_T = k.transpose(2, 3)
        ## (B, h, N, D//h) x (B, h, D//h, N) -> (B, h, N, N) 
        dots = (q @ k_T) / self.sqrt_dh
        ## Softmax along the columns
        attn = F.softmax(dots, dim=-1)
        ## Dropout
        attn = self.attn_drop(attn)
        # Weighted sum [Equation (8)]
        ## (B, h, N, N) x (B, h, N, D//h) -> (B, h, N, D//h) 
        out = attn @ v
        ## (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.transpose(1, 2)
        ## (B, N, h, D//h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        # Output layer [Equation (10)]
        ## (B, N, D) -> (B, N, D) 
        out = self.w_o(out) 
        return out

mhsa = MultiHeadSelfAttention()
out = mhsa(z_0)  # z_0 is from section 2-2 z_0=input_layer(x), shape (B, N, D)

# Check that it is (2, 5, 384)(=(B, N, D)) 
print(out.shape)



# ----------------------------
# 2-5 Encoder
# ----------------------------
print("=======2-5 Encoder=======")

class VitEncoderBlock(nn.Module): 
    def __init__(self, emb_dim:int=384, head:int=8, hidden_dim:int=384*4, dropout: float=0.):
        """
        Arguments:
            emb_dim: Length of the embedded vector
            head: Number of heads
            hidden_dim: Length of the intermediate vector in the MLP of the Encoder Block
                        Following the original paper, four times emb_dim is set as the default value
            dropout: Dropout rate
        """
        super(VitEncoderBlock, self).__init__()
        # First Layer Normalization [Section 2-5-2]
        self.ln1 = nn.LayerNorm(emb_dim)
        # MHSA [Section 2-4-7]
        self.msa = MultiHeadSelfAttention(
            emb_dim=emb_dim, head=head,
            dropout=dropout,
        )
        # Second Layer Normalization [Section 2-5-2] 
        self.ln2 = nn.LayerNorm(emb_dim)
        # MLP [Section 2-5-3]
        self.mlp = nn.Sequential( 
            nn.Linear(emb_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, emb_dim), 
            nn.Dropout(dropout)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """ 
        Arguments:
            z: Input to the Encoder Block. Shape: (B, N, D)
                B: Batch size, N: Number of tokens, D: Length of the vector

        Returns:
            out: Output of the Encoder Block. Shape: (B, N, D). [Equation (10)]
                B: Batch size, N: Number of tokens, D: Length of the embedded vector 
        """
        # First half of the Encoder Block [Equation (12)] 
        out = self.msa(self.ln1(z)) + z
        # Second half of the Encoder Block [Equation (13)] 
        out = self.mlp(self.ln2(out)) + out 
        return out

vit_enc = VitEncoderBlock()
z_1 = vit_enc(z_0)  # z_0 is from section 2-2 z_0=input_layer(x), shape (B, N, D)

# Check that it is (2, 5, 384)(=(B, N, D)) 
print(z_1.shape)



# ----------------------------
# 2-6 Implementation of ViT
# ----------------------------
print("=======2-6 Implementation of ViT=======")

class Vit(nn.Module): 
    def __init__(self, in_channels:int=3, num_classes:int=10, emb_dim:int=384, num_patch_row:int=2, image_size:int=32, num_blocks:int=7, head:int=8, hidden_dim:int=384*4, dropout:float=0.):
        """
        Arguments:
            in_channels: Number of channels in the input image
            num_classes: Number of classes for image classification
            emb_dim: Length of the embedded vector
            num_patch_row: Number of patches on one side
            image_size: Size of one side of the input image. It is assumed that the height and width of the input image are the same
            num_blocks: Number of Encoder Blocks
            head: Number of heads
            hidden_dim: Length of the intermediate vector in the MLP of the Encoder Block
            dropout: Dropout rate
        """
        super(Vit, self).__init__()
        # Input Layer [Section 2-3] 
        self.input_layer = VitInputLayer(
            in_channels, 
            emb_dim, 
            num_patch_row, 
            image_size)

        # Encoder. Multiple stages of Encoder Blocks. [Section 2-5] 
        self.encoder = nn.Sequential(*[
            VitEncoderBlock(
                emb_dim=emb_dim,
                head=head,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_blocks)])

        # MLP Head [Section 2-6-1] 
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Input image to ViT. Shape: (B, C, H, W)
                B: Batch size, C: Channel count, H: Height, W: Width

        Returns:
            out: Output of ViT. Shape: (B, M). [Equation (10)]
                B: Batch size, M: Number of classes 
        """
        # Input Layer [Equation (14)]
        ## (B, C, H, W) -> (B, N, D)
        ## N: Number of tokens (= number of patches + 1), D: Length of the vector 
        out = self.input_layer(x)
        
        # Encoder [Equations (15) and (16)]
        ## (B, N, D) -> (B, N, D)
        out = self.encoder(out)

        # Extract only the class token
        ## (B, N, D) -> (B, D)
        cls_token = out[:,0]

        # MLP Head [Equation (17)]
        ## (B, D) -> (B, M)
        pred = self.mlp_head(cls_token)
        return pred

num_classes = 10
batch_size, channel, height, width = 2, 3, 32, 32
x = torch.randn(batch_size, channel, height, width)
vit = Vit(in_channels=channel, num_classes=num_classes) 
pred = vit(x)

# Check that it is (2, 10)(=(B, M)) 
print(pred.shape)