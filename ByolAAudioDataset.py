from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize

from AugmentationPipeline import AugmentationPipeline
from utils import init_logger


class BYOLAudioDataset(Dataset):
    """
    Audio dataset for BYOL-A self-supervised learning.

    Generates pairs of augmented spectrograms from audio files.

    Parameters
    ----------
    root_dir : Path
        Directory containing audio files
    spectrogram_params : dict[str, Any]
        Parameters for librosa melspectrogram generation
    augmentation_pipeline : AugmentationPipeline
        Pipeline for spectrogram augmentations
    spectrogram_dimensions : tuple[int, int], default=(384, 384)
        Target (height, width) for final spectrograms
    training_length_in_seconds : float, default=5.0
        Target duration for audio clips
    file_extension : str, default='.wav'
        Audio file extension to search for in the root_dir
    """

    def __init__(self,
                 root_dir: Path,
                 spectrogram_params: dict[str, Any],
                 augmentation_pipeline: AugmentationPipeline,
                 spectrogram_dimensions: tuple[int, int] = (384, 384),
                 training_length_in_seconds: float = 5.0,
                 file_extension: str = '.wav') -> None:
        """Initialize the dataset"""
        super().__init__()
        self.logger = init_logger('Dataset')
        self.root_dir = root_dir
        self.logger.info(
            f'Looking for audio files with the following extension: {file_extension}')
        self.files = list(root_dir.rglob(f'*{file_extension}'))
        self.n = len(self.files)
        self.logger.info(f'Found {self.n} Files')
        self.augmentation_pipeline = augmentation_pipeline
        self.spectrogram_params = spectrogram_params
        self.rng = np.random.default_rng(42)
        self.resize = Resize(
            (spectrogram_dimensions[0], spectrogram_dimensions[1]))
        self.audio_length_values = int(
            self.spectrogram_params['sr'] * training_length_in_seconds)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate pair of augmented spectrograms from single audio file.

        Parameters
        ----------
        index : int
            Index of audio file to load

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Two augmented views of a snippet of the audio file
        """
        audio_path = self.files[index]
        audio, _ = librosa.load(audio_path, sr=self.spectrogram_params['sr'])
        audio = self.adjust_audio_length(audio, self.audio_length_values)
        spectrogram = torch.from_numpy(
            self.generate_spectrogram(audio)).float().unsqueeze(0)
        spectrogram_1 = self.resize(self.augmentation_pipeline(spectrogram))
        spectrogram_2 = self.resize(self.augmentation_pipeline(spectrogram))

        return spectrogram_1, spectrogram_2

    def generate_spectrogram(self, waveform: np.ndarray, top_db: int = 80) -> np.ndarray:
        """
        Convert waveform to mel spectrogram with optional dB scaling.

        Parameters
        ----------
        waveform : np.ndarray
            Waveform representation of the audio file
        top_db : int, default=80
            Threshold for db scaling

        Returns
        -------
        np.ndarray
            Spectrogram
        """
        spec = librosa.feature.melspectrogram(
            y=waveform, **self.spectrogram_params)
        if self.spectrogram_params['power'] >= 2.0:
            spec = librosa.power_to_db(spec, ref=np.max, top_db=top_db)
        return spec

    def adjust_audio_length(self, audio: np.ndarray, audio_length_values: int) -> np.ndarray:
        """
        Ensure audio clip has exact length via padding/cropping.

        Parameters
        ----------
        audio : np.ndarray
            Input audio signal
        audio_length_values : int
            Target number of samples

        Returns
        -------
        np.ndarray
            Padded or cropped audio array
        """
        if len(audio) < audio_length_values:
            padded_audio = np.zeros(audio_length_values)
            missing_padding = audio_length_values - len(audio)
            missing_padding_before = int(missing_padding * self.rng.random())
            padded_audio[missing_padding_before: missing_padding_before +
                         len(audio)] = audio
            return padded_audio
        start_idx = int((len(audio) - audio_length_values)
                        * self.rng.random())
        cropped_audio = audio[start_idx: start_idx + audio_length_values]
        return cropped_audio

    def __len__(self) -> int:
        """Return total number of audio files.

        Returns
        -------
        int
            Length of dataset
        """
        return self.n

    def get_data_loader(self,
                        batch_size: int = 16,
                        num_workers: int = 12,
                        shuffle: bool = True,
                        drop_last: bool = True,
                        pin_memory: bool = True,
                        prefetch_factor: int = 2,
                        persistent_workers: bool = True) -> DataLoader:
        """
        Create DataLoader for this dataset.

        Parameters
        ----------
        batch_size : int, default=16
            Samples per batch
        num_workers, int, default=12
            Number of workers for data loading
        shuffle : bool, default=True
            Shuffle dataset
        drop_last : bool, default=True
            Discard last incomplete batch
        pin_memory : bool, default=True
            Use pinned memory for faster GPU transfer
        prefetch_factor : int, default=2
            Batches to prefetch per worker
        persistent_workers : bool, default=True
            Maintain workers between epochs

        Returns
        -------
        DataLoader
            DataLoader for this dataset
        """
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle,
                          drop_last=drop_last,
                          pin_memory=pin_memory,
                          prefetch_factor=prefetch_factor,
                          persistent_workers=persistent_workers)
