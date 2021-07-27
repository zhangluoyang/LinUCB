import os
import torch
import datetime
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from Loss import BasicLoss
from typing import Dict, List
from Metrics import BasicMetrics
from Layer import BasicLayer
from Data import Data

__version__ = "1.5.3"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

layer_modules = (torch.nn.MultiheadAttention,)


def to_device(tensor_dict: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    """
    将数据放置指定设备上面
    :param tensor_dict:
    :param device:
    :return:
    """
    tensor_dict_ = {}
    for key in tensor_dict.keys():
        tensor_dict_[key] = torch.tensor(tensor_dict[key]).to(device)
    return tensor_dict_


class Model(torch.nn.Module):

    # print time bar...
    @staticmethod
    def print_bar():
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "=" * 80 + "%s" % nowtime)

    def __init__(self, net: BasicLayer):
        super(Model, self).__init__()
        self.net = net

        self.history = {}
        self.loss_funcs: List[BasicLoss] = NotImplemented
        self.metrics_funcs: List[BasicMetrics] = NotImplemented
        self.device = NotImplemented
        self.early_stop = NotImplemented
        self.optimizer = NotImplemented
        self.tensor_names: List[str] = NotImplemented

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.net:
            return self.net.forward(tensor_dict=tensor_dict)
        else:
            raise NotImplementedError

    def compile(self, optimizer=None,
                device=None,
                early_stop: int = 5):
        """
        :param optimizer:  优化器
        :param device:  设备
        :param early_stop:  提前停止的条件
        :return:
        """
        self.loss_funcs = self.net.build_loss_layers()
        self.metrics_funcs = self.net.build_metric_layers()
        self.early_stop = early_stop
        self.optimizer = optimizer if optimizer else torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)
        self.device = device if torch.cuda.is_available() else None
        if self.device:
            self.to(self.device)

    def train_step(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        :param tensor_dict:
        :return:
        """
        self.train()
        self.optimizer.zero_grad()
        predictions = self.forward(tensor_dict)
        output_dict = dict(tensor_dict, **predictions)
        train_metrics = {}

        loss_list: List[torch.Tensor] = []
        for loss_func in self.loss_funcs:
            loss = loss_func.forward(output_dict)
            train_metrics[loss_func.name] = loss.item()
            loss_list.append(loss)
        for metric_func in self.metrics_funcs:
            metrics = metric_func.forward(output_dict)
            train_metrics[metric_func.name] = metrics.item()
        # 所有损失函数之和
        loss_sum = sum(loss_list)
        train_metrics["loss"] = loss_sum.item()
        loss_sum.backward()
        # update parameters
        self.optimizer.step()
        self.optimizer.zero_grad()

        return train_metrics

    @torch.no_grad()
    def evaluate_step(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        predictions = self.forward(tensor_dict)
        output_dict = dict(tensor_dict, **predictions)
        val_metrics = {}
        with torch.no_grad():
            loss_list: List[torch.Tensor] = []
            for loss_func in self.loss_funcs:
                loss = loss_func.forward(output_dict)
                val_metrics[loss_func.name] = loss.item()
                loss_list.append(loss)
            for metric_func in self.metrics_funcs:
                metrics = metric_func.forward(output_dict)
                val_metrics[metric_func.name] = metrics.item()
            # 所有损失函数之和
            loss_sum = sum(loss_list)
            val_metrics["loss"] = loss_sum.item()
        return val_metrics

    def fit(self, epochs: int, dl_train: Data, dl_val: Data = None,
            batch_sie: int = 128,
            log_step_freq=1, check_path: str = None):

        print("Start Training ...")
        Model.print_bar()

        dl_val = dl_val if dl_val else []

        # 上一步验证集最小的index
        last_min_eval_loss = 1e10
        last_min_eval_index = 0
        for epoch in range(1, epochs + 1):
            dl_train.shuffle()
            # 1，training loop -------------------------------------------------
            train_metrics_sum, step = {}, 0
            while True:
                batch_data = dl_train.next(batch_size=batch_sie)
                if batch_data is None:
                    break
                tensor_dict = to_device(tensor_dict=batch_data, device=self.device)
                step = step + 1
                train_metrics = self.train_step(tensor_dict=tensor_dict)
                for name, metric in train_metrics.items():
                    train_metrics_sum[name] = train_metrics_sum.get(name, 0.0) + metric
                if step % log_step_freq == 0:
                    logs = {"step": step}
                    logs.update({k: round(v / step, 3) for k, v in train_metrics_sum.items()})
                    print(logs)
            for name, metric_sum in train_metrics_sum.items():
                self.history[name] = self.history.get(name, []) + [metric_sum / step]

            # 2，validate loop -------------------------------------------------
            val_metrics_sum, step = {}, 0
            if dl_val is not None:
                dl_val.shuffle()
                while True:
                    batch_data = dl_val.next(batch_size=batch_sie)
                    if batch_data is None:
                        break
                    tensor_dict = to_device(tensor_dict=batch_data, device=self.device)
                    step = step + 1
                    val_metrics = self.evaluate_step(tensor_dict=tensor_dict)
                    for name, metric in val_metrics.items():
                        val_metrics_sum[name] = val_metrics_sum.get(name, 0.0) + metric
                for name, metric_sum in val_metrics_sum.items():
                    self.history[name] = self.history.get(name, []) + [metric_sum / step]

            if check_path is not None:
                if not os.path.exists(check_path):
                    print("mask dir:{0}".format(check_path))
                    os.makedirs(check_path)
                torch.save(self.net.state_dict(), "{0}/epoch_{1}.pth".format(check_path, epoch))
                remove_path = "{0}/epoch_{1}.pth".format(check_path, epoch - self.early_stop - 1)
                if os.path.exists(remove_path):
                    os.remove(remove_path)
                    print("remove:{0}".format(remove_path))

            if self.early_stop is not None and dl_val is not None:
                epoch_eval_loss = val_metrics_sum["loss"] / step
                print("check early stop")
                print("epoch_eval_loss:{0}, epoch_eval_loss:{1}, last_min_eval_index:{2}, epoch:{3}".format(
                    epoch_eval_loss,
                    last_min_eval_loss,
                    last_min_eval_index,
                    epoch))
                # 连续early_stop 次结果没有提升 则退出
                if epoch_eval_loss > last_min_eval_loss and epoch - last_min_eval_index >= self.early_stop:
                    print("early stop ....")
                    break
                if epoch_eval_loss < last_min_eval_loss:
                    last_min_eval_loss = epoch_eval_loss
                    last_min_eval_index = epoch
            # 3，print logs -------------------------------------------------
            infos = {"epoch": epoch}
            infos.update({k: round(self.history[k][-1], 3) for k in self.history})
            tb = PrettyTable()
            tb.field_names = infos.keys()
            tb.add_row(infos.values())
            print("\n", tb)
            Model.print_bar()

        # 仅保留一个模型文件
        for epoch in range(1, epochs + 1):
            path = "{0}/epoch_{1}.pth".format(check_path, epoch)
            if epoch != last_min_eval_index:
                if os.path.exists(path):
                    os.remove(path)
            else:
                os.rename(path, "{0}/best.pth".format(check_path))
                print("Finished Training...")

        return pd.DataFrame(self.history)
