import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader



def validate_model(model: nn.Module):
    if not isinstance(model, nn.Module) or model is None:
        raise TypeError(f"'model' must be a 'nn.Module' sub-class. But 'model' is {type(model).__name__} instead")

def validate_optimizer(optimizer: optim.Optimizer):
    if not isinstance(optimizer, optim.Optimizer) or optimizer is None:
        raise TypeError(f"'optimizer' must be an 'optim.Optimizer' sub-class. But 'optimizer' is {type(optimizer).__name__} instead")

def validate_lossfn(loss_fn: nn.modules.loss._Loss):
    if not isinstance(loss_fn, nn.modules.loss._Loss) or loss_fn is None:
        raise TypeError(f"'loss_fn' must be an 'nn.modules.loss._Loss' sub-class. But 'loss_fn' is {type(loss_fn).__name__} instead")

def validate_data_loaders(train_loader: DataLoader, test_loader: DataLoader):
    if not isinstance(train_loader, DataLoader) or train_loader is None:
        raise TypeError(f"'train_loader' must be a 'DataLoader' instance. But 'train_loader' is {type(train_loader).__name__} instead")
    
    if not isinstance(test_loader, DataLoader) or test_loader is None:
        raise TypeError(f"'test_loader' must be a 'DataLoader' instance. But 'test_loader' is {type(test_loader).__name__} instead")


def validate_trainer_attributes(trainer):

    validate_model(trainer.model)
    
    validate_optimizer(trainer.optimizer)
    
    validate_lossfn(trainer.loss_fn)
    
    validate_data_loaders(trainer.train_loader, trainer.test_loader)
