from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class AugmentationPipeline(nn.Module):

    def __init__(self,
                 initial_norm_mean: float = 0.0,
                 initial_norm_std: float = 1.0,
                 mixup_alpha: float = 0.4,
                 memory_size: int = 2048) -> None:
        super().__init__()
        self.initial_norm_mean = initial_norm_mean
        self.initial_norm_std = initial_norm_std
        self.mixup_alpha = mixup_alpha
        self.cached_audio = deque(maxlen=memory_size)
        self.running_stats = dict(
            current_mean=initial_norm_mean,
            current_std=initial_norm_std,
            count=1
        )

    def running_norm(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        if update:
            count = self.running_stats['count']
            new_mean = (self.running_stats['current_mean']
                        * count + x.mean().item()) / (count + 1)
            new_std = (self.running_stats['current_std']
                       * count + x.std().item()) / (count + 1)

            self.running_stats['current_mean'] = new_mean
            self.running_stats['current_std'] = new_std
            self.running_stats['count'] = count + 1

        return (x - self.running_stats['current_mean']) / self.running_stats['current_std']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.running_norm(x)
        if len(self.cached_audio) == 0:
            x_mix = x
        else:
            idx = torch.randint(0, len(self.cached_audio), (1,)).item()
            x_mix = self.cached_audio[idx].to(x.device)
        mixed = self.mixup(x, x_mix, (0, self.mixup_alpha))
        self.cached_audio.append(x.detach().cpu())

        cropped = self.random_resize_crop(mixed)
        faded = self.random_linear_fade(cropped)
        output = self.running_norm(faded, False)

        return output

    @staticmethod
    def mixup(
            spec_1: torch.Tensor,
            spec_2: torch.Tensor,
            lambda_range: tuple[float, float] = (0, 0.3)) -> torch.Tensor:
        lmbda = (lambda_range[1] - lambda_range[0]) * \
            torch.rand(1, device=spec_1.device) + lambda_range[0]
        mixed = (1 - lmbda) * spec_1.exp() + lmbda * spec_2.exp()
        return mixed.log()

    @staticmethod
    def random_linear_fade(
            spectrogram: torch.Tensor,
            uniform_dist_range: tuple[float, float] = (-1, 1)) -> torch.Tensor:
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
        crop = virtual_crop_area[:, i:i+crop_h, j:j+crop_w]
        crop = F.interpolate(crop.unsqueeze(0), size=(
            FR, T), mode=interpolation, align_corners=True)
        crop = crop.squeeze(0)

        return crop.to(spectrogram.dtype)
