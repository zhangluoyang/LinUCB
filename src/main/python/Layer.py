import torch
import torch.nn as nn
from config import Config
from typing import List, Dict
from Loss import BasicLoss, MseLoss
from Metrics import BasicMetrics, Accuracy


class BasicLayer(nn.Module):

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplemented

    def build_metric_layers(self) -> List[BasicMetrics]:
        raise NotImplemented

    def build_loss_layers(self) -> List[BasicLoss]:
        raise NotImplemented


class MovieRecommendation(BasicLayer):

    def __init__(self, config: Config):
        super(MovieRecommendation, self).__init__()
        self.config = config

        self.user_embedding_layer = nn.Embedding(num_embeddings=self.config.user_num,
                                                 embedding_dim=self.config.user_id_embedding_size)

        self.item_embedding_layer = nn.Embedding(num_embeddings=self.config.item_num,
                                                 embedding_dim=self.config.item_id_embedding_size)

        user_in_features = self.config.user_id_embedding_size + 1 + self.config.occupation_num

        self.user_layer = nn.Linear(in_features=user_in_features, out_features=self.config.user_feature_dim)

        item_in_features = self.config.item_id_embedding_size + self.config.style_num

        self.item_layer = nn.Linear(in_features=item_in_features, out_features=self.config.item_feature_dim)

    def user_feature(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        user_id = tensor_dict["user_id"]
        occupation = tensor_dict["occupation"]
        age = tensor_dict["age"]

        user_embedding = self.user_embedding_layer(user_id)
        user_embedding = torch.cat(tensors=(user_embedding, occupation, age), dim=-1)
        user_feature = self.user_layer(user_embedding)
        user_feature = torch.relu(user_feature)
        return user_feature

    def item_feature(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        item_id = tensor_dict["item_id"]
        detail = tensor_dict["detail"]

        item_embedding = self.item_embedding_layer(item_id)
        item_embedding = torch.cat(tensors=(item_embedding, detail), dim=-1)
        item_feature = self.item_layer(item_embedding)
        item_feature = torch.relu(item_feature)
        return item_feature

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        user_feature = self.user_feature(tensor_dict=tensor_dict)
        item_feature = self.item_feature(tensor_dict=tensor_dict)
        logits = torch.sum(user_feature * item_feature, dim=-1)
        tensor_dict["logits"] = logits
        return tensor_dict

    def build_loss_layers(self) -> List[BasicLoss]:
        loss = MseLoss(tensor_name="logits", target_tensor_name="rating")
        return [loss]

    def build_metric_layers(self) -> List[BasicMetrics]:
        acc = Accuracy(tensor_name="logits", target_name="rating")
        return []
