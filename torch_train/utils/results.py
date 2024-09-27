from dataclasses import dataclass, field
from abc import ABC

@dataclass
class TrainigOutputsResults:

    train_loss: list[float] = field(default_factory= lambda: [])
    test_loss: list[float] = field(default_factory= lambda: [])

    train_acc: list[float] = field(default_factory= lambda: [])
    test_acc: list[float] = field(default_factory= lambda: [])

    def reset(self):
        self.train_loss.clear()
        self.test_loss.clear()
        self.train_acc.clear()
        self.test_acc.clear()

    def set_train_loss(self, train_loss: float):
        self.train_loss.append(
            train_loss
        )

    def set_test_loss(self, test_loss: float):
        self.test_loss.append(
            test_loss
        )

    def set_train_acc(self, train_acc: float):
        self.train_acc.append(train_acc)

    def set_test_acc(self, test_acc: float):
        self.test_acc.append(test_acc)


    