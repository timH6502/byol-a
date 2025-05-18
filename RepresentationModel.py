import torch
import torch.nn as nn


class RepresentationModel(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 input_shape: tuple[int, int],
                 projector_dim: int = 256,
                 hidden_dim: int = 4096,
                 dropout_p: float = 0.2,
                 include_predictor: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        dummy_input = torch.zeros(1, 1, *input_shape)
        self.encoder_dim = self.backbone(dummy_input).shape[-1]

        self.projector = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_p),
            nn.GELU(),
            nn.Linear(hidden_dim, projector_dim)
        )

        self.include_predictor = include_predictor
        if include_predictor:
            self.predictor = nn.Sequential(
                nn.Linear(projector_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, projector_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        p = self.projector(z)
        if self.include_predictor:
            q = self.predictor(p)
            return z, p, q
        return z, p
