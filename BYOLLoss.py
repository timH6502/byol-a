import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class BYOLLoss(_Loss):
    """
    BYOL loss function.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize BYOL loss module."""
        super().__init__(*args, **kwargs)

    def forward(self, q_z_phi: torch.Tensor, z_xi: torch.Tensor) -> torch.Tensor:
        """
        Compute BYOL loss between online predictions and target projections.

        Parameters
        ----------
        q_z_phi : torch.Tensor
            Online network predictions
            Shape: (batch_size, projection_dim)
        z_xi : torch.Tensor
            Target network projections
            Shape: (batch_size, projection_dim)
        """
        q_z_phi = F.normalize(q_z_phi, p=2, dim=-1)
        z_xi = F.normalize(z_xi, p=2, dim=-1)
        cos_sim = (q_z_phi * z_xi).sum(dim=-1)
        loss = 2 - 2 * cos_sim.mean()
        return loss
