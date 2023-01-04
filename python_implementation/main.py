import os
import ssl
import time

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from eval import plot_losses, show_batch_images
from LeNet5 import LeNet5, load_model, save_model
from train import train_epochs


ssl._create_default_https_context = ssl._create_unverified_context  # for downloading dataset

DEVICE = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.01
BATCH_SIZE = 32
N_EPOCHS = 15
CLASSES = (
  'plane', 'car', 'bird', 'cat', 'deer',
  'dog', 'frog', 'horse', 'ship', 'truck'
)


def main():

  # define transforms
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  # download and create datasets
  train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
  )
  val_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
  )

  # define the data loaders
  train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
  )
  val_loader = DataLoader(
    dataset=val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False
  )

  # initialize model and optimizer
  model = LeNet5(len(CLASSES))
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

  # # train model
  # model, _, (train_losses, val_losses) = train_epochs(model, criterion, optimizer, train_loader, val_loader, N_EPOCHS, DEVICE)
  # plot_losses(train_losses, val_losses)
  # save_model(model, 'weights')
  model = load_model(model, 'weights')

  # get inputs and outputs for a single batch
  data_iter = iter(train_loader)
  inputs, labels = next(data_iter)
  _, outputs = model(inputs)
  preds = outputs.max(dim=1).indices

  # # show images of batch
  # show_batch_images(inputs)
  # print(' '.join('%5s' % CLASSES[preds[j]] for j in range(labels.shape[0])))
  # print(' '.join('%5s' % CLASSES[labels[j]] for j in range(labels.shape[0])))

  # select sample image
  idx = (preds == labels).max(dim=0).indices
  image = inputs[idx]
  image_unprocessed = image / 2 + 0.5
  label = CLASSES[labels[idx]]

  # save sample image
  file_path = os.path.join('test_images', label)
  torchvision.utils.save_image(image, f'{file_path}_processed.png')
  torchvision.utils.save_image(image_unprocessed, f'{file_path}_raw.png')

  # time forward pass on test image
  image = image.unsqueeze(dim=0)
  start_time = time.time()
  _, output = model(image)
  duration = time.time() - start_time
  prediction = CLASSES[output.max(dim=1).indices]
  print(f"Took {duration} seconds to make prediction \"{prediction}\" for image. ")


if __name__ == '__main__':
  main()