import torch
import torch.nn as nn
import torch.nn.functional as F


class ReluOutputModel(nn.Module):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        n_hidden: int,
    ):
        """ReLU output model from Toy Models of Superposition

        ref: https://transformer-circuits.pub/2022/toy_model/index.html#motivation
        code source: https://github.com/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb

        Args:
            n_instances: number of models to train
            n_features: number of input features
            n_hidden: number of hidden units
            feature_probability: _description_. Defaults to None.
            importance: _description_. Defaults to None.
        """
        super().__init__()
        self.W = nn.Parameter(torch.empty((n_instances, n_features, n_hidden)))
        nn.init.xavier_normal_(self.W)
        self.b_final = nn.Parameter(torch.zeros((n_instances, n_features)))

    def forward(self, features):
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        hidden = torch.einsum("...if,ifh->...ih", features, self.W)
        out = torch.einsum("...ih,ifh->...if", hidden, self.W)
        out = out + self.b_final
        out = F.relu(out)
        return out
