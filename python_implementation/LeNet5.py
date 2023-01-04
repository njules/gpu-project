import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):

  def __init__(self, n_classes: int):
    super().__init__()

    # input: 32x32x3
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
    # 28x28x6
    self.pool1 = nn.AvgPool2d(kernel_size=2)
    # 14x14x6
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
    # 10x10x16
    self.pool2 = nn.AvgPool2d(kernel_size=2)
    # 5x5x16
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
    # 1x1x120
    self.fc1 = nn.Linear(in_features=120, out_features=84)
    self.fc2 = nn.Linear(in_features=84, out_features=n_classes)

  def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = inputs
    x = self.conv1(x)
    x = nn.Sigmoid()(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = nn.Sigmoid()(x)
    x = self.pool2(x)
    x = self.conv3(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = nn.Sigmoid()(x)
    x = self.fc2(x)
    logits = x
    predictions = F.softmax(logits, dim=1)
    return logits, predictions


def save_model(model: nn.Module, path: str) -> None:
  """Saves the network parameters as multiple .npy files, one per layer.
  """

  state_dict = model.state_dict()
  for layer in state_dict:
    file_path = os.path.join(path, layer)
    np.save(file_path, state_dict[layer].numpy())


def load_model(model: nn.Module, path: str) -> nn.Module:
  state_dict = model.state_dict()

  for layer in state_dict:
    file_path = os.path.join(path, layer) + '.npy'
    print(f"deb: {path}, {layer}, {file_path} ")
    state_dict[layer] = torch.Tensor(np.load(file_path))

  model.load_state_dict(state_dict)
  return model

