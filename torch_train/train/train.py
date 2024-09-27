from torch.utils.data import DataLoader
import torch, torch.nn as nn

from tqdm import tqdm

from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from ..trainer import TorchTrainer, TorchTrainerConfigs


def train_model(
    model: torch.nn.Module, 
    loss_fn: torch.nn.modules.loss._Loss,
    train_loader: DataLoader,
    train_step: Callable[['TorchTrainer', torch.Tensor], None],
    configs: 'TorchTrainerConfigs', 
):
    steps: int = 1
    for epoch in (bar:= tqdm(range(configs.num_iters))):
        total_loss = 0
        for step, (inputs, labels) in enumerate(train_loader):
            batch = (
                inputs.to(configs.device),
                labels.to(configs.device)
            )
            outs = train_step(model=model, loss_fn=loss_fn, batch=batch, steps=steps)

            steps += 1

            total_loss += outs.loss.item()



        bar.set_description(f"Loss: {total_loss / len(train_loader):.4}")

