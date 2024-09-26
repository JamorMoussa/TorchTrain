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
        
        validate_trainer_attributes(trainer=self)

        if num_iters is not None:
            self.configs.num_iters = num_iters

        train_model(
            trainer= self,
            train_step= self.train_step_func, 
            configs= self.configs
        )

