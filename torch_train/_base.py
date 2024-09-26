import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
import functools 


class BaseTainerConfigs:
    pass 


class BaseTrainer(ABC):


    def __init__(self, configs: BaseTainerConfigs):
        
        self.configs: BaseTainerConfigs = configs

        self.model: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.loss_fn: nn.modules.loss._Loss = None 
        self.train_loader: DataLoader = None
        self.test_loader: DataLoader = None 

        self.train_step_func: Callable = None
        self.test_step_func: Callable = None


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

    def train_step(self, train_step_wrapped_func):
        @functools.wraps(train_step_wrapped_func)
        def wrapper(*args, **kwargs):
            train_step_wrapped_func(self, *args, **kwargs)

        self.train_step_func = wrapper
        return wrapper

    def test_step(self, test_step_wrapped_func):
        @functools.wraps(test_step_wrapped_func)
        def wrapper(*args, **kwargs):
            test_step_wrapped_func(self, *args, **kwargs)
        
        self.test_step_func = wrapper
        return wrapper

    @abstractmethod
    def train(self, num_iters: int):
        ...
