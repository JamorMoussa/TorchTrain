from .utils import OutputsForClassification


from torch.utils.data import DataLoader
import torch, torch.nn as nn
import torch.optim as optim

from abc import ABC, abstractmethod
from typing import Callable
import functools 



class BaseTainerConfigs(ABC):
    
    @staticmethod
    @abstractmethod
    def defaults():
        ...


class BaseTrainer(ABC):

    def __init__(self, configs: BaseTainerConfigs):
        super(BaseTrainer, self).__init__()
        
        self.configs: BaseTainerConfigs = configs

        self.model: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.loss_fn: nn.modules.loss._Loss = None 
        self.train_loader: DataLoader = None
        self.test_loader: DataLoader = None 

        self._train_test_step: Callable = None

        self._run_each_epoch: Callable = None 


    def set_model(
        self, model: torch.nn.Module
    ) -> None:
        self.model = model.to(self.configs.device)
    
    def set_optimizer(
        self, optimizer: optim.Optimizer
    ) -> None:
        self.optimizer = optimizer

    def set_lossfn(
        self, loss_fn: nn.modules.loss._Loss
    ) -> None:
        self.loss_fn = loss_fn

    def set_data_loaders(
        self, train_loader: DataLoader, test_loader: DataLoader
    ) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader


    def train_test_step(self, train_test_func):

        # TODO: #3 No need for a test_step wrapper
        
        @functools.wraps(train_test_func)
        def wrapper(*args, **kargs):

            outs: OutputsForClassification  = train_test_func(*args, **kargs)

            if self.model.training:
                self.optimizer.zero_grad()

                outs.loss.backward()

                self.optimizer.step()

            return outs 

        self._train_test_step = wrapper
        return wrapper

    def run_each_epoch(self, run_each_epoch):

        @functools.wraps(run_each_epoch)
        def wrapper(*args, **kwargs):
            
            return run_each_epoch(*args, **kwargs)
        
        self._run_each_epoch = wrapper
        return wrapper


    @abstractmethod
    def train(self, num_iters: int):
        ...
