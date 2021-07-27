from typing import Union, Dict

import torch
import torch.nn as nn
from torch.nn import functional as F


class BasicLoss(nn.Module):
    """
    损失函数
    """

    def __init__(self, tensor_name: Union[str, None],
                 target_tensor_name: Union[str, None],
                 name: str,
                 eof: float = 1.0):
        """
        :param tensor_name: tensor的名称
        :param target_tensor_name: 目标 tensor的名称
        :param name: 损失的名称
        :param eof: 损失的权重
        """
        super(BasicLoss, self).__init__()
        self.tensor_name = tensor_name
        self.target_tensor_name = target_tensor_name
        self.name = name
        self.eof = eof

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplemented


class CrossEntropyLoss(BasicLoss):
    """
    交叉熵代价函数
    """

    def __init__(self, tensor_name: Union[str, None],
                 target_tensor_name: Union[str, None],
                 name: str = "loss",
                 eof: float = 1.0):
        super(CrossEntropyLoss, self).__init__(tensor_name=tensor_name, target_tensor_name=target_tensor_name,
                                               name=name, eof=eof)

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(tensor_dict[self.tensor_name], tensor_dict[self.target_tensor_name])


class MseLoss(BasicLoss):

    def __init__(self, tensor_name: Union[str, None],
                 target_tensor_name: Union[str, None],
                 name: str = "loss",
                 eof: float = 1.0):
        super(MseLoss, self).__init__(tensor_name=tensor_name, target_tensor_name=target_tensor_name,
                                      name=name, eof=eof)

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.mse_loss(tensor_dict[self.tensor_name], tensor_dict[self.target_tensor_name])
