from ..utils.results import TrainigOutputsResults 

from torch.utils.data import DataLoader
import torch, torch.nn as nn

from tqdm import tqdm

from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from ..trainer import TorchTrainerConfigs


def train_model(
    model: torch.nn.Module, 
    loss_fn: torch.nn.modules.loss._Loss,
    train_loader: DataLoader,
    test_loader: DataLoader,
    train_test_step: Callable,
    run_each_epoch: Callable,
    configs: 'TorchTrainerConfigs', 
):
    steps: int = 1

    results_outer = TrainigOutputsResults()

    for epoch in (bar:= tqdm(range(configs.num_iters))):
        train_loss = 0
        
        model.train()
        for step, (inputs, labels) in enumerate(train_loader):
            
            batch = (
                inputs.to(configs.device),
                labels.to(configs.device)
            )
            outs = train_test_step(model=model, loss_fn=loss_fn, batch=batch, steps=steps)

            steps += 1

            train_loss += outs.loss.item()

        with torch.no_grad():
            test_loss = 0

            model.eval()
            for v_step, (v_inputs, v_labels) in enumerate(test_loader):
                v_batch = (
                    v_inputs.to(configs.device), v_labels.to(configs.device)
                )

                v_outs = train_test_step(model=model, loss_fn=loss_fn, batch=v_batch, steps=steps)

                test_loss += v_outs.loss.item()

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        results_outer.set_train_loss(train_loss= train_loss)
        results_outer.set_test_loss(test_loss= test_loss)

        run_each_epoch()

        bar.set_description(f"Epoch {epoch+1}/{configs.num_iters} | Train loss: {train_loss:.4} | Test loss: {test_loss:.4}")

    return results_outer