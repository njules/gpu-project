import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision


def compute_accuracy(
  model: nn.Module,
  data_loader: DataLoader,
  device: torch.device
) -> float:
  """Compute accuracy of model over dataset.
  """

  total_pred = 0
  correct_pred = 0 

  with torch.no_grad():
    for inputs, labels in data_loader:

      inputs = inputs.to(device)
      labels = labels.to(device)

      _, outputs = model(inputs)
      predicted_labels = torch.max(outputs, dim=1).indices

      total_pred += labels.size(0)
      correct_pred += (predicted_labels == labels).sum()

  accuracy = correct_pred / total_pred
  return accuracy


def compute_loss(
  model: nn.Module,
  val_loader: DataLoader,
  criterion: nn.Module,
  device: torch.device
) -> float:
  """Compute average loss over validation dataset.
  """

  running_loss = 0

  with torch.no_grad():
    for inputs, labels in val_loader:

      inputs = inputs.to(device)
      labels = labels.to(device)

      logits, _ = model(inputs)
      loss = criterion(logits, labels)
      running_loss += loss * labels.size(0)

  total_loss = running_loss / len(val_loader.dataset)
  return total_loss


def show_batch_images(
  images: torch.Tensor
):

  image_grid = torchvision.utils.make_grid(images)
  image_grid = image_grid / 2 + 0.5
  npimg = image_grid.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


def plot_losses(train_losses: List[float], val_losses: List[float]) -> None:
  """Plot training and validation loss curves.
  """
  
  # temporarily change the style of the plots to seaborn 
  plt.style.use('seaborn')

  train_losses = np.array(train_losses) 
  val_losses = np.array(val_losses)

  fig, ax = plt.subplots(figsize = (8, 4.5))

  ax.plot(train_losses, color='blue', label='Training loss') 
  ax.plot(val_losses, color='red', label='Validation loss')
  ax.set(
    title="Loss over epochs", 
    xlabel='Epoch',
    ylabel='Loss'
  ) 
  ax.legend()
  fig.show()
  
  # change the plot style to default
  plt.style.use('default')
