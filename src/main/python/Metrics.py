import torch
from typing import Dict, List


class BasicMetrics(object):

    def __init__(self, tensor_name: str, name: str):
        self.tensor_name = tensor_name
        self.name = name

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplemented


class Accuracy(BasicMetrics):
    def __init__(self, tensor_name: str, target_name: str, name: str = "acc", threshold: float = 0.5):
        super(Accuracy, self).__init__(tensor_name=tensor_name, name=name)
        self.threshold = threshold
        self.target_name = target_name

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        :param tensor_dict
        :return:
        """
        logits = tensor_dict[self.tensor_name]
        y_pred = torch.argmax(logits, dim=1)
        y_true = tensor_dict[self.target_name]
        return torch.mean((y_pred == y_true).float())
