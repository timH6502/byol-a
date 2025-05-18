from pathlib import Path

import timm
import torch

from torch.amp import GradScaler

from AugmentationPipeline import AugmentationPipeline
from ByolAAudioDataset import BYOLAudioDataset
from BYOLATrainer import BYOLATrainer
from BYOLLoss import BYOLLoss
from RepresentationModel import RepresentationModel
from TauScheduler import TauScheduler
from utils import set_seeds


if __name__ == '__main__':

    set_seeds()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 50
    use_amp = True
    virtual_batch_size = 2048
    batch_size = 32
    accumulation_steps = virtual_batch_size // batch_size

    spectrogram_shape = (256, 256)

    spectrogram_params = dict(
        sr=32000,
        n_fft=2048,
        hop_length=512,
        win_length=1536,
        n_mels=160,
        fmin=10,
        fmax=16000,
        power=2.0,
        norm='slaney',
        pad_mode='reflect',
        center=True,
        window='hann',
        htk=False,
    )

    backbone_params = dict(
        pretrained=True,
        drop_rate=0.1,
        drop_path_rate=0.1,
        model_name='tf_efficientnet_b3.ns_jft_in1k',
        num_classes=None,
        in_chans=1,
    )

    online_model = RepresentationModel(
        backbone=timm.create_model(**backbone_params),
        input_shape=spectrogram_shape,
        include_predictor=True)

    target_model = RepresentationModel(
        backbone=timm.create_model(**backbone_params),
        input_shape=spectrogram_shape)

    target_model.backbone.load_state_dict(online_model.backbone.state_dict())
    target_model.projector.load_state_dict(online_model.projector.state_dict())

    for param in target_model.parameters():
        param.requires_grad_(False)

    dataset = BYOLAudioDataset(
        root_dir=Path('./sound_data'),
        spectrogram_params=spectrogram_params,
        augmentation_pipeline=AugmentationPipeline(),
        spectrogram_dimensions=spectrogram_shape)

    tau_scheduler = TauScheduler(
        int(len(dataset) * epochs / (accumulation_steps * batch_size)), 0.996)

    optimizer = torch.optim.AdamW(online_model.parameters(
    ), lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=10),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs-10, eta_min=1e-6)
        ],
        milestones=[10]
    )

    loss_function = BYOLLoss()
    grad_scaler = GradScaler()

    trainer = BYOLATrainer(
        online_model=online_model,
        target_model=target_model,
        tau_scheduler=tau_scheduler,
        optimizer=optimizer,
        loss_function=loss_function,
        data_set=dataset,
        lr_scheduler=lr_scheduler,
        grad_scaler=grad_scaler,
        device=device,
        use_amp=use_amp,
        accumulation_steps=accumulation_steps,
        batch_size=batch_size
    )

    trainer.train(epochs)
