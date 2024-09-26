import torch_train as ttn
import torch 

configs = ttn.TorchTrainerConfigs(
    device= torch.device("cuda")
)

trainer = ttn.TorchTrainer(configs= configs)


@trainer.train_step
def train_step(trainer: ttn.TorchTrainer, batch: torch.Tensor):
    print(trainer.__class__.__name__)
    print(trainer.configs.device)

@trainer.test_step
def test_step(trainer: ttn.TorchTrainer, batch: torch.Tensor):
    print(trainer.model)


trainer.train()