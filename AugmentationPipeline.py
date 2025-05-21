from collections import deque
from multiprocessing import Value, Manager

import torch
import torch.nn as nn
import torch.nn.functional as F


class AugmentationPipeline(nn.Module):
    """
    Audio spectrogram augmentation pipeline for BYOL-A self-supervised learning.

    Parameters
    ----------
    initial_norm_mean : float, default=0.0
        Initial value for running mean of normalization
    initial_norm_std : float, default=1.0
        Initial value for running standard deviation
    mixup_alpha : float, default=0.4
        Maximum mixup coefficient
    memory_size : int, defualt=2048
        Size of memory bank for mixup candidates
    """

    def __init__(self,
                 initial_norm_mean: float = 0.0,
                 initial_norm_std: float = 1.0,
                 mixup_alpha: float = 0.4,
                 memory_size: int = 2048,
                 post_norm_ema_decay: float = 0.995) -> None:
        """
        Initialize augmentation pipeline with normalization and mixup memory.
        """
        super().__init__()
        self.initial_norm_mean = initial_norm_mean
        self.initial_norm_std = initial_norm_std
        self.mixup_alpha = mixup_alpha
        self.cached_audio = deque(maxlen=memory_size)
        self.post_norm_ema_decay = post_norm_ema_decay
        self.pre_norm_running_stats = {
            'current_mean': Value('d', initial_norm_mean),
            'current_std': Value('d', initial_norm_std),
            'count': Value('i', 1),
            'lock': Manager().Lock()
        }
        self.post_norm_running_stats = {
            'current_mean': Value('d', initial_norm_mean),
            'current_std': Value('d', initial_norm_std),
            'lock': Manager().Lock()
        }

    def running_norm(self, x: torch.Tensor, pre_norm: bool = True) -> torch.Tensor:
        """
        Apply running normalization with optional statistic update.

        Parameters
        ----------
        x : torch.Tensor
            Input spectrogram of shape without batch dimension -> (C, Freq, Time)
        pre_norm : bool, optional
            Whether pre- or post-norm is used.

        Returns
        -------
        torch.Tensor
            Normalized spectrogram
        """
        if pre_norm:
            stats = self.pre_norm_running_stats
            with stats['lock']:
                count = stats['count'].value
                new_mean = (stats['current_mean'].value *
                            count + x.mean().item()) / (count + 1)
                new_std = (stats['current_std'].value *
                           count + x.std().item()) / (count + 1)
                stats['current_mean'].value = new_mean
                stats['current_std'].value = new_std
                stats['count'].value = count + 1
        else:
            stats = self.post_norm_running_stats
            with stats['lock']:
                current_mean = stats['current_mean'].value
                current_std = stats['current_std'].value

                new_mean = (self.post_norm_ema_decay * current_mean +
                            (1 - self.post_norm_ema_decay) * x.mean().item())
                new_std = (self.post_norm_ema_decay * current_std +
                           (1 - self.post_norm_ema_decay) * x.std().item())

                stats['current_mean'].value = new_mean
                stats['current_std'].value = new_std

        return (x - stats['current_mean'].value) / \
            stats['current_std'].value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply full augmentation pipeline to input spectrogram.

        Parameters
        ----------
        x : torch.Tensor
            Raw input spectrogram of shape without batch dimension -> (C, Freq, Time)

        Returns
        -------
        torch.Tensor
            Augmented spectrogram
        """
        x = self.running_norm(x, pre_norm=True)
        if len(self.cached_audio) == 0:
            x_mix = x
        else:
            idx = torch.randint(0, len(self.cached_audio), (1,)).item()
            x_mix = self.cached_audio[idx].to(x.device)
        mixed = self.mixup(x, x_mix, (0, self.mixup_alpha))
        self.cached_audio.append(x.detach().cpu())

        cropped = self.random_resize_crop(mixed)
        faded = self.random_linear_fade(cropped)
        output = self.running_norm(faded, pre_norm=False)

        return output

    @staticmethod
    def mixup(
            spec_1: torch.Tensor,
            spec_2: torch.Tensor,
            lambda_range: tuple[float, float] = (0, 0.3)) -> torch.Tensor:
        """
        Mix two spectrograms.

        Parameters
        ----------
        spec_1 : torch.Tensor
            First spectrogram
        spec_2 : torch.Tensor
            Second spectrogram
        lambda_range : tuple[float, float], default=(0.0, 0.3)
            Range for mixup coefficient. Sampled from a unifor distribution.

        Returns
        -------
        torch.Tensor
            Mixed spectrograms
        """
        lmbda = (lambda_range[1] - lambda_range[0]) * \
            torch.rand(1, device=spec_1.device) + lambda_range[0]
        mixed = (1 - lmbda) * spec_1.exp() + lmbda * spec_2.exp()
        return mixed.log()

    @staticmethod
    def random_linear_fade(
            spectrogram: torch.Tensor,
            uniform_dist_range: tuple[float, float] = (-1, 1)) -> torch.Tensor:
        """
        Add time-linear gain curve to spectrogram

        Parameters
        ----------
        spectrogram : torch.Tensor
            Input spectrogram
        uniform_dist_range : tuple[float, float], default=(-1, 1)
            Slope of the curve

        Returns
        -------
        torch.Tensor
            Augmented spectrogram
        """
        _, _, T = spectrogram.shape
        alpha, beta = (uniform_dist_range[1] - uniform_dist_range[0]) * torch.rand(
            2, device=spectrogram.device) + uniform_dist_range[0]
        t = torch.linspace(0, 1, steps=T, device=spectrogram.device)
        fade = alpha + (beta - alpha) * t
        fade = fade.unsqueeze(0).unsqueeze(1)
        augmented_spec = spectrogram + fade
        return augmented_spec

    @staticmethod
    def random_resize_crop(
            spectrogram: torch.Tensor,
            virtual_crop_scale: tuple[float, float] = (1.0, 1.5),
            freq_scale: tuple[float, float] = (0.6, 1.5),
            time_scale: tuple[float, float] = (0.6, 1.5),
            interpolation: str = 'bicubic') -> torch.Tensor:
        """
        Random spatiotemporal crop with resizing to original dimensions

        Parameters
        ----------
        spectrogram : torch.Tensor
            Input spectrogram
        virtual_crop_scale : tuple[float, float], default=(1.0, 1.5)
            Scale range for virtual crop area
        freq_scale : tuple[float, float], default=(0.6, 1.5)
            Frequency axis scaling range
        time_scale : tuple[float, float], default=(0.6, 1.5)
            Time axis scaling range
        interpolation : str, default='bicubic'
            Interpolation method for resize

        Returns
        -------
        torch.Tensor
            Augmented spectrogram

        """
        C, FR, T = spectrogram.shape
        virtual_crop_h = int(FR * virtual_crop_scale[0])
        virtual_crop_w = int(T * virtual_crop_scale[1])
        virtual_crop_area = torch.zeros(
            (C, virtual_crop_h, virtual_crop_w), device=spectrogram.device, dtype=torch.float)
        x = (virtual_crop_w - T) // 2
        y = (virtual_crop_h - FR) // 2
        virtual_crop_area[:, y: y + FR, x: x + T] = spectrogram
        freq_factor, time_factor = (
            torch.rand(2, device=spectrogram.device) * (torch.tensor(
                [freq_scale[1] - freq_scale[0], time_scale[1] - time_scale[0]], device=spectrogram.device))
            + torch.tensor([freq_scale[0], time_scale[0]],
                           device=spectrogram.device)
        )
        crop_h = int(torch.clamp(freq_factor * FR,
                     min=1, max=virtual_crop_h).item())
        crop_w = int(torch.clamp(time_factor * T,
                     min=1, max=virtual_crop_w).item())
        max_i = max(virtual_crop_h - crop_h, 1)
        max_j = max(virtual_crop_w - crop_w, 1)
        i = torch.randint(0, max_i, (1,), device=spectrogram.device).item()
        j = torch.randint(0, max_j, (1,), device=spectrogram.device).item()
        crop = virtual_crop_area[:, i:i + crop_h, j:j + crop_w]
        crop = F.interpolate(crop.unsqueeze(0), size=(
            FR, T), mode=interpolation, align_corners=True)
        crop = crop.squeeze(0)

        return crop.to(spectrogram.dtype)
