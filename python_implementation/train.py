from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from eval import compute_accuracy, compute_loss


def train_one_epoch(train_loader, model, criterion, optimizer, device):
  """
  Function for the training step of the training loop
  """

  running_loss = 0
  
  for inputs, labels in train_loader:

    optimizer.zero_grad()
    
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Forward pass
    predictions, _ = model(inputs) 
    loss = criterion(predictions, labels) 
    running_loss += loss.item() * inputs.size(0)

    # Backward pass
    loss.backward()
    optimizer.step()
    
  epoch_loss = running_loss / len(train_loader.dataset)
  return model, optimizer, epoch_loss


def train_epochs(
  model: nn.Module,
  criterion: nn.Module,
  optimizer: optim.Optimizer,
  train_loader: DataLoader,
  val_loader: DataLoader,
  epochs: int,
  device: torch.device,
  print_every=1
) -> Tuple[nn.Module, optim.Optimizer, Tuple[List[float], List[float]]]:
  """Function defining the entire training loop
  """
  
  # set objects for storing metrics
  train_losses = []
  val_losses = []

  # Train model
  for epoch in range(0, epochs):

    model.train()
    model, optimizer, train_loss = train_one_epoch(
      train_loader,
      model,
      criterion,
      optimizer,
      device
    )
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
      val_loss = compute_loss(model, val_loader, criterion, device)
      val_losses.append(val_loss)

      if epoch % print_every == (print_every - 1):
        train_acc = compute_accuracy(model, train_loader, device=device)
        val_acc = compute_accuracy(model, val_loader, device=device)

        print(
          f'{datetime.now().time().replace(microsecond=0)} --- '
          f'Epoch: {epoch}\t'
          f'Train loss: {train_loss:.4f}\t'
          f'Validation loss: {val_loss:.4f}\t'
          f'Train accuracy: {100 * train_acc:.2f}\t'
          f'Validation accuracy: {100 * val_acc:.2f}'
        )

  return model, optimizer, (train_losses, val_losses)