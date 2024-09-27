from ._base import BaseTrainer, BaseTainerConfigs
from .utils.validate import validate_trainer_attributes
from .train import train_model

from dataclasses import dataclass, field
from typing import Self

import torch


@dataclass
class TorchTrainerConfigs(BaseTainerConfigs):

    device: torch.DeviceObjType = field(default_factory= lambda: torch.device("cpu"))
    num_iters: int = field(default_factory= lambda: 5)

    @staticmethod
    def defaults() -> Self:
        return TorchTrainerConfigs()
    


class TorchTrainer(BaseTrainer):

    def __init__(self, configs: TorchTrainerConfigs = None) -> None:
        configs = configs if configs is not None else TorchTrainerConfigs.defaults()
        super(TorchTrainer, self).__init__(configs = configs)


    def train(self, num_iters: int = None):

        ## TODO: #2 make train_step optinal
        # User may not specify a train_step, so the trainer must use a default 
        # train_step/test_step for more custimisation.
        
        validate_trainer_attributes(trainer=self)

        if num_iters is not None:
            self.configs.num_iters = num_iters

        train_model(
            model= self.model,
            loss_fn= self.loss_fn,
            train_loader= self.train_loader,
            train_step= self.train_step_func, 
            configs= self.configs
        )

