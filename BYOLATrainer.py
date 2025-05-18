from pathlib import Path

import numpy as np
import torch

from torch.amp import GradScaler
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.modules.loss import _Loss
from tqdm import tqdm

from TauScheduler import TauScheduler
from RepresentationModel import RepresentationModel
from ByolAAudioDataset import BYOLAudioDataset
from utils import init_logger


class BYOLATrainer:
    """
    BYOL-A trainer.

    Parameters
    ----------
    online_model : RepresentationModel
        Online network that includes backbone, projector and predictor
    target_model : RepresentationModel
        Target network that includes backbone and projector
    tau_scheduler : TauScheduler
        Scheduler for target network EMA decau parameter tau
    optimizer : Optimizer
        Optimizer for the online network
    loss_function : _Loss
        BYOL loss
    data_set : BYOLAudioDataset
        Audio dataset providing two augmented versions of an audio snippet
    lr_scheduler : _LRScheduler
        Learning rate scheduler
    grad_scaler : GradScaler
        Gradient scaler for mixed precision training
    device : torch.device
        Target device for training (CPU or GPU)
    use_amp : bool, default=False
        Enable automatic mixed precision
    accumulation_steps : int, default=1
        Number of gradient accumulation steps
    batch_size : int, default=64
        True batch size
    """

    def __init__(self,
                 online_model: RepresentationModel,
                 target_model: RepresentationModel,
                 tau_scheduler: TauScheduler,
                 optimizer: Optimizer,
                 loss_function: _Loss,
                 data_set: BYOLAudioDataset,
                 lr_scheduler: _LRScheduler,
                 grad_scaler: GradScaler,
                 device: torch.device,
                 use_amp: bool = False,
                 accumulation_steps: int = 1,
                 batch_size: int = 64,
                 save_path: Path = Path('./models')) -> None:
        self.online_model = online_model.to(device)
        self.target_model = target_model.to(device)
        self.tau_scheduler = tau_scheduler
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.data_loader = data_set.get_data_loader(batch_size=batch_size)
        self.lr_scheduler = lr_scheduler
        self.grad_scaler = grad_scaler
        self.use_amp = use_amp
        self.accumulation_steps = accumulation_steps
        self.device = device
        self.save_path = save_path

        if not self.save_path.exists():
            self.save_path.mkdir()

        self.logger = init_logger('Trainer')

        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        self.logger.info(f'Trainer initialized. Training on {self.device}.')

    def _train_one_epoch(self) -> float:
        """
        Train models for one epoch.

        Returns
        -------
        float
            Mean loss value across all batches in the epoch
        """
        losses = np.zeros(len(self.data_loader))
        progress_bar = tqdm(self.data_loader, desc="Training",
                            leave=True, total=len(self.data_loader))
        self.online_model.train(True)
        for i, (spectrograms_1, spectrograms_2) in enumerate(progress_bar):
            spectrograms_1 = spectrograms_1.to(self.device)
            spectrograms_2 = spectrograms_2.to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                _, _, predictions_online_1 = self.online_model(spectrograms_1)
                with torch.no_grad():
                    _, predictions_target_1 = self.target_model(spectrograms_2)
                loss_1 = self.loss_function(
                    predictions_online_1, predictions_target_1)

                _, _, predictions_online_2 = self.online_model(spectrograms_2)
                with torch.no_grad():
                    _, predictions_target_2 = self.target_model(spectrograms_1)
                loss_2 = self.loss_function(
                    predictions_online_2, predictions_target_2)

                loss = (loss_1 + loss_2) / 2

            self.grad_scaler.scale(loss / self.accumulation_steps).backward()

            if (i + 1) % self.accumulation_steps == 0:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
                self._update_target()

            losses[i] = loss.item()
            progress_bar.set_postfix(
                loss=losses[i], tau=self.tau_scheduler.get_current_tau())

        if (i + 1) % self.accumulation_steps != 0:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()
            self._update_target()

        self.online_model.train(False)
        return losses.mean()

    def train(self, epochs: int) -> np.ndarray:
        """
        Run complete training cycle for specified number of epochs.

        Parameters
        ----------
        epochs : int
            Number of epochs to train

        Returns
        -------
        np.ndarray
            Array of average loss values per epoch
        """
        losses = np.zeros(epochs)
        self.optimizer.zero_grad()
        for i in range(epochs):
            self.logger.info(f'{"-" * 100}')
            self.logger.info(
                f'Epoch: {i + 1}\tLearning rate: {self.lr_scheduler.get_last_lr()[-1]}')
            losses[i] = self._train_one_epoch()
            self.logger.info(
                f'Average loss: {losses[i]}')
            self.lr_scheduler.step()
            self._save_models()
        return losses

    def _update_target(self) -> None:
        """
        Update target network parameters using exponential moving average.
        """
        with torch.no_grad():
            current_tau = self.tau_scheduler.step()
            for t, o in zip(self.target_model.backbone.parameters(), self.online_model.backbone.parameters()):
                t.data = current_tau * t.data + (1 - current_tau) * o.data
            for t, o in zip(self.target_model.projector.parameters(), self.online_model.projector.parameters()):
                t.data = current_tau * t.data + (1 - current_tau) * o.data

    def _save_models(self) -> None:
        """
        Save current state of online and target models to disk.
        """
        torch.save(self.online_model.state_dict(),
                   self.save_path / 'online_model.pth')
        torch.save(self.target_model.state_dict(),
                   self.save_path / 'target_model.pth')
