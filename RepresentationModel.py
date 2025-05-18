import torch
import torch.nn as nn


class RepresentationModel(nn.Module):
    """
    BYOL-A representation model combining backbone encoder with projection head and optional prediction head.

    Parameters
    ----------
    backbone : nn.Module
        Encoding network
    input_shape : tuple[int, int]
        Expected input spectrogram dimensions (height, width)
    projector_dim : int, default=256
        Dimension of projected latent space
    hidden_dim : int, default=4096
        Hidden layer dimension in projector/predictor
    dropout_p : float, defualt=0.0
        Dropout probability in projector
    include_predictor : bool, default=False
        Whether to include predictor MLP
    """

    def __init__(self,
                 backbone: nn.Module,
                 input_shape: tuple[int, int],
                 projector_dim: int = 256,
                 hidden_dim: int = 4096,
                 dropout_p: float = 0.0,
                 include_predictor: bool = False) -> None:
        """Initialize representation model."""

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
        """
        Forward pass through network components.

        Parameters
        ----------
        x : torch.Tensor
            Input spectrogram tensor of shape (batch_size, 1, freq_bins, time_steps)

        Returns
        -------
        torch.Tensor or tuple
            - With predictor: (backbone_out, projector_out, predictor_out)
            - Without predictor: (backbone_out, projector_out)
        """
        z = self.backbone(x)
        p = self.projector(z)
        if self.include_predictor:
            q = self.predictor(p)
            return z, p, q
        return z, p
