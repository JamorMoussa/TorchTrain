from ..metrics import Metric

import torch 

from dataclasses import dataclass, field


@dataclass
class OutputsForClassification:

    loss: torch.nn.modules.loss._Loss = field(default_factory= lambda: None)
    outputs: torch.Tensor = field(default_factory= lambda: None)

    metrics: list[Metric] = field(default_factory= lambda: [])

