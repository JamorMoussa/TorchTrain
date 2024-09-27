import torch_train as ttn
from torch_train.utils import OutputsResults

import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


X = torch.randn(10000, 3)
y = torch.mm(X, torch.Tensor([1, 2, 3]).view(-1, 1)) + 0.5 * torch.randn(10000, 1)


dataset = TensorDataset(X, y)

train_loader = DataLoader(dataset= dataset, shuffle=True, batch_size=32)

model = nn.Sequential(
    nn.Linear(3, 2),
    nn.ReLU(),
    nn.Linear(2, 1)
)

loss_fn = nn.MSELoss()

opt = optim.SGD(model.parameters(), lr=0.01)

configs = ttn.TorchTrainerConfigs(
    device= torch.device("cpu"),
    num_iters=100
)

trainer = ttn.TorchTrainer(configs= configs)

trainer.set_model(model=model)
trainer.set_lossfn(loss_fn=loss_fn)
trainer.set_optimizer(optimizer=opt)
trainer.set_data_loaders(train_loader=train_loader, test_loader=train_loader)


@trainer.train_step
def train_step(model, loss_fn, batch, steps):

    inputs, labels = batch

    outputs = model(inputs)

    loss = loss_fn(outputs, labels)

    return OutputsResults(
        outputs= outputs, loss=loss
    )

trainer.train(num_iters=100)