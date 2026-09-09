import math
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding with learnable/registered buffer of shape (1, max_len, d_model).
    """
    def __init__(self, d_model: int = 256, max_len: int = 65, dropout: float = 0.4):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ASLTransformer(nn.Module):
    """
    Exact ASLTransformer architecture matching ASL_95class_75_72pct_TEST_best.pth:
    - Input dimension: 696
    - d_model: 256
    - nhead: 4
    - num_layers: 4
    - dim_feedforward: 512
    - dropout: 0.4
    - activation: GELU
    - Learned CLS token
    - Sequence length: 65 (1 CLS token + 64 temporal frames)
    - Output classes: 95
    """
    def __init__(
        self,
        input_dim: int = 696,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.4,
        num_classes: int = 95
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=65, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, 64, 696)
        batch_size = x.shape[0]
        x = self.input_proj(x)  # (Batch, 64, 256)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (Batch, 1, 256)
        x = torch.cat((cls_tokens, x), dim=1)  # (Batch, 65, 256)
        x = self.pos_encoding(x)  # (Batch, 65, 256)
        x = self.encoder(x)  # (Batch, 65, 256)
        cls_out = x[:, 0]  # (Batch, 256)
        logits = self.classifier(cls_out)  # (Batch, 95)
        return logits

# Global model singleton
_MODEL: Optional[ASLTransformer] = None
_CLASS_NAMES: Optional[Dict[int, str]] = None
_DEVICE: Optional[torch.device] = None

def load_asl_model(checkpoint_path: str = "model/ASL_95class_75_72pct_TEST_best.pth") -> Tuple[ASLTransformer, Dict[int, str], torch.device]:
    """
    Loads the trained ASLTransformer checkpoint once and returns (model, class_names, device).
    """
    global _MODEL, _CLASS_NAMES, _DEVICE
    if _MODEL is not None:
        return _MODEL, _CLASS_NAMES, _DEVICE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _DEVICE = device

    if not os.path.isabs(checkpoint_path):
        # Resolve against project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        potential_path = os.path.join(base_dir, checkpoint_path)
        if os.path.exists(potential_path):
            checkpoint_path = potential_path

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Extract class names
    if "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
    else:
        raise ValueError("Checkpoint does not contain 'class_names'")

    num_classes = len(class_names)
    model = ASLTransformer(input_dim=696, num_classes=num_classes)
    
    state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _MODEL = model
    _CLASS_NAMES = class_names

    print(f"ASL model loaded successfully on {device.type.upper()}")
    print(f"Classes: {num_classes}")
    print(f"Input dimension: 696 (348 base + 348 velocity)")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return _MODEL, _CLASS_NAMES, _DEVICE
