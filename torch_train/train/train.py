from tqdm import tqdm
import torch

from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from ..trainer import TorchTrainer, TorchTrainerConfigs


def train_model(
    trainer: 'TorchTrainer',
    train_step: Callable[['TorchTrainer', torch.Tensor], None],
    configs: 'TorchTrainerConfigs', 
):
    
    for epoch in tqdm(range(configs.num_iters)):

        for step, (inputs, labels) in trainer.train_loader:
            batch = (
                inputs.to(configs.device),
                labels.to(configs.device)
            )
            train_step(trainer, batch)