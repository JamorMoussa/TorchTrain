from ._base import BaseTrainer, BaseTainerConfigs
from .utils.validate import validate_trainer_attributes

from dataclasses import dataclass, field
import torch


@dataclass
class TorchTrainerConfigs(BaseTainerConfigs):

    device: torch.DeviceObjType = field(default_factory= lambda: torch.device("cpu"))

    @staticmethod
    def defaults(): return TorchTrainerConfigs()
    


class TorchTrainer(BaseTrainer):

    def __init__(self, configs: TorchTrainerConfigs = None) -> None:
        configs = configs if configs is not None else TorchTrainerConfigs.defaults()
        super(TorchTrainer, self).__init__(configs = configs)


    def train(self, num_iters: int = 100):
        # validate trainer attributes: 
        # validate_trainer_attributes(trainer=self)


        self.train_step_func()
        self.test_step_func()

