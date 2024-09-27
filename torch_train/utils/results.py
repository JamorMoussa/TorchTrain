import torch 

from dataclasses import dataclass, field


@dataclass
class OutputsResults:

    outputs: torch.Tensor = field(default_factory= lambda: None)
    loss: torch.nn.modules.loss._Loss = field(default_factory= lambda: None)